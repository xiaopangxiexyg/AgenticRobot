"""Headless deterministic evaluation for a temporary PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from sim.mujoco.collision_safety import CollisionEventTracker
from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import (
    CubeLiftTaskConfig,
    is_cube_lift_success_reason,
)
from sim.mujoco.tasks.cube_lift.learning import (
    build_policy_action_transform,
    encode_policy_observation,
)

from .action_diagnostics import ActionDiagnostics
from .checkpoint import build_policy_from_checkpoint, check_task_config_compatibility
from .policy import resolve_device
from .pose_target_diagnostics import PoseTargetDiagnostics
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
    parser.add_argument("--num-envs", type=int, default=4)
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
    task_config = CubeLiftTaskConfig.load(args.task_config)
    action_transform = build_policy_action_transform(task_config)
    if (
        policy.action_dim != action_transform.policy_dim
        or policy.environment_action_dim != action_transform.task_dim
        or tuple(policy.active_action_indices) != action_transform.task_indices
    ):
        raise ValueError(
            "Checkpoint policy action dimensions do not match the task-owned "
            "latent-to-pose transform."
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
    blocking_body_pairs: Counter[str] = Counter()
    blocking_categories: Counter[str] = Counter()
    blocking_category_projection_fraction_sum: Counter[str] = Counter()
    blocking_category_rejections: Counter[str] = Counter()
    phase_blocking_categories: Counter[str] = Counter()
    phase_actions: Counter[str] = Counter()
    phase_projected_actions: Counter[str] = Counter()
    minimum_safety_distance = float("inf")
    hard_penetrations = 0
    clearance_violations = 0
    collision_preventive_events = 0
    collision_preventive_blocked_fraction_sum = 0.0
    collision_preventive_streak_steps_max = 0
    collision_preventive_event_categories: Counter[str] = Counter()
    collision_actual_forbidden_contacts = 0
    collision_hard_violations = 0
    tracking_error_sum = 0.0
    tracking_error_samples = 0
    tracking_error_max = 0.0
    gripper_tracking_error_sum = 0.0
    gripper_tracking_error_max = 0.0
    action_diagnostics = ActionDiagnostics()
    pose_target_diagnostics = PoseTargetDiagnostics()
    collision_event_tracker = CollisionEventTracker(args.num_envs)
    try:
        observation, _infos = env.reset()
        max_steps = args.episodes * 600
        for _ in range(max_steps):
            actor_observation = encode_policy_observation(
                task_config,
                observation,
            )
            with torch.no_grad():
                action, _log_prob = policy.predict_action(
                    torch.as_tensor(
                        normalizer.normalize(actor_observation),
                        device=device,
                    ),
                    deterministic=not args.stochastic,
                )
                if policy.action_dim == 1:
                    action[:, 0] = torch.clamp(
                        action[:, 0]
                        + torch.as_tensor(offset_by_env, device=device),
                        -1.0,
                        1.0,
                    )
                latent_action = action.cpu().numpy().astype(
                    np.float32, copy=False
                )
                action_scale = action_transform.action_scale(observation)
                environment_action = action_transform.transform(
                    latent_action,
                    observation=observation,
                )
                action_diagnostics.update(
                    latent_action,
                    environment_action,
                    action_scale,
                )
            observation, reward, terminated, truncated, infos = env.step(
                environment_action
            )
            evaluated_actions += args.num_envs
            done = np.logical_or(terminated, truncated)
            transition_infos = tuple(
                info.get("final_info", info) if done[env_id] else info
                for env_id, info in enumerate(infos)
            )
            collision_events = collision_event_tracker.observe(
                preventive_intervention=np.asarray(
                    [
                        bool(
                            info.get(
                                "safety_intervention",
                                info.get("safety_modified", False),
                            )
                        )
                        for info in transition_infos
                    ],
                    dtype=bool,
                ),
                projection_alpha=np.asarray(
                    [
                        float(info.get("safety_projection_alpha", 1.0))
                        for info in transition_infos
                    ],
                    dtype=np.float32,
                ),
                preventive_category=np.asarray(
                    [
                        info.get("safety_blocking_category")
                        for info in transition_infos
                    ],
                    dtype=object,
                ),
                actual_forbidden_contact=np.asarray(
                    [
                        bool(
                            info.get(
                                "safety_actual_forbidden_contact",
                                info.get("termination_reason")
                                == "robot_environment_penetration",
                            )
                        )
                        for info in transition_infos
                    ],
                    dtype=bool,
                ),
                actual_category=np.asarray(
                    [
                        info.get("safety_actual_collision_category")
                        for info in transition_infos
                    ],
                    dtype=object,
                ),
                hard_violation=np.asarray(
                    [
                        bool(
                            info.get(
                                "safety_actual_hard_violation",
                                info.get("termination_reason")
                                == "robot_environment_penetration",
                            )
                        )
                        for info in transition_infos
                    ],
                    dtype=bool,
                ),
                episode_done=done,
            )
            pose_target_diagnostics.update(
                np.asarray(
                    [
                        np.nan
                        if info.get("pose_target_pre_action_translation_error_m") is None
                        else float(info["pose_target_pre_action_translation_error_m"])
                        for info in transition_infos
                    ],
                    dtype=np.float64,
                ),
                projection_alpha=np.asarray(
                    [
                        float(info.get("safety_projection_alpha", 1.0))
                        for info in transition_infos
                    ],
                    dtype=np.float64,
                ),
                preventive_intervention=np.asarray(
                    [
                        bool(
                            info.get(
                                "safety_intervention",
                                info.get("safety_modified", False),
                            )
                        )
                        for info in transition_infos
                    ],
                    dtype=bool,
                ),
                preventive_onset=collision_events.preventive_onset,
            )
            collision_preventive_events += int(
                np.count_nonzero(collision_events.preventive_onset)
            )
            collision_preventive_blocked_fraction_sum += float(
                np.sum(collision_events.preventive_blocked_fraction)
            )
            collision_preventive_streak_steps_max = max(
                collision_preventive_streak_steps_max,
                int(np.max(collision_events.preventive_streak_steps)),
            )
            collision_actual_forbidden_contacts += int(
                np.count_nonzero(collision_events.actual_forbidden_contact)
            )
            collision_hard_violations += int(
                np.count_nonzero(collision_events.hard_violation)
            )
            for category in collision_events.preventive_category[
                collision_events.preventive_onset
            ]:
                collision_preventive_event_categories[
                    "unknown" if category is None else str(category)
                ] += 1
            for info in transition_infos:
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
                safety_modified = bool(info.get("safety_modified", False))
                projected_actions += int(safety_modified)
                rejected_actions += int(
                    float(info.get("safety_projection_alpha", 1.0)) <= 1e-8
                )
                if bool(info.get("post_contact_phase_active", False)):
                    phase = "post_contact"
                elif float(
                    info.get("gripper_bilateral_inner_contact_score", 0.0)
                ) >= 0.5:
                    phase = "bilateral_inner_contact"
                else:
                    phase = "pre_contact"
                phase_actions[phase] += 1
                phase_projected_actions[phase] += int(safety_modified)
                if safety_modified:
                    body1 = info.get("safety_blocking_body1")
                    body2 = info.get("safety_blocking_body2")
                    pair = " <-> ".join(
                        sorted(
                            (
                                "<unknown>" if body1 is None else str(body1),
                                "<unknown>" if body2 is None else str(body2),
                            )
                        )
                    )
                    blocking_body_pairs[pair] += 1
                    category = str(
                        info.get("safety_blocking_category") or "unknown"
                    )
                    blocking_categories[category] += 1
                    blocking_category_projection_fraction_sum[category] += (
                        1.0
                        - float(info.get("safety_projection_alpha", 1.0))
                    )
                    blocking_category_rejections[category] += int(
                        float(info.get("safety_projection_alpha", 1.0))
                        <= 1e-8
                    )
                    phase_blocking_categories[f"{phase}: {category}"] += 1
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
            for env_id in np.flatnonzero(done):
                if completed_by_env[env_id] < target_episodes_per_env:
                    episode_return = float(returns[env_id])
                    episode_length = int(lengths[env_id])
                    success = int(
                        is_cube_lift_success_reason(
                            transition_infos[env_id].get("termination_reason")
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
        "evaluation_seed": int(args.seed),
        "task_objective": task_config.task_objective,
        "task_config_semantic_sha256": task_config.source_sha256,
        "cube_spawn_xy_jitter_m": list(
            task_config.cube_spawn_xy_jitter_m
        ),
        "cube_spawn_yaw_jitter_rad": (
            task_config.cube_spawn_yaw_jitter_rad
        ),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "deterministic": not args.stochastic,
        "policy_action_dim": policy.action_dim,
        "environment_action_dim": policy.environment_action_dim,
        "active_action_indices": list(policy.active_action_indices),
        "policy_action_components": list(action_transform.policy_components),
        "policy_action_transform_id": action_transform.transform_id,
        "policy_action_transform_hash": action_transform.semantic_content_hash,
        "task_action_layout": task_config.action_layout,
        **action_diagnostics.summary(),
        **pose_target_diagnostics.summary(),
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
        "projected_action_count": projected_actions,
        "fully_rejected_action_count": rejected_actions,
        "projected_action_rate_by_phase": {
            phase: phase_projected_actions[phase] / max(count, 1)
            for phase, count in sorted(phase_actions.items())
        },
        "action_count_by_phase": dict(sorted(phase_actions.items())),
        "safety_blocking_body_pair_counts": dict(
            sorted(
                blocking_body_pairs.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "safety_blocking_category_counts": dict(
            sorted(blocking_categories.items())
        ),
        "safety_blocking_category_action_rates": {
            category: count / max(evaluated_actions, 1)
            for category, count in sorted(blocking_categories.items())
        },
        "safety_blocking_category_projection_fraction_mean": {
            category: blocking_category_projection_fraction_sum[category]
            / max(count, 1)
            for category, count in sorted(blocking_categories.items())
        },
        "safety_blocking_category_rejection_rates": {
            category: blocking_category_rejections[category] / max(count, 1)
            for category, count in sorted(blocking_categories.items())
        },
        "safety_blocking_phase_category_counts": dict(
            sorted(phase_blocking_categories.items())
        ),
        "minimum_robot_environment_distance_m": (
            None if not np.isfinite(minimum_safety_distance) else minimum_safety_distance
        ),
        "hard_penetration_count": hard_penetrations,
        "clearance_violation_count": clearance_violations,
        "collision_preventive_event_count": collision_preventive_events,
        "collision_preventive_event_rate": (
            collision_preventive_events / max(evaluated_actions, 1)
        ),
        "collision_preventive_blocked_fraction_mean": (
            collision_preventive_blocked_fraction_sum
            / max(evaluated_actions, 1)
        ),
        "collision_preventive_streak_steps_max": (
            collision_preventive_streak_steps_max
        ),
        "collision_preventive_event_category_counts": dict(
            sorted(collision_preventive_event_categories.items())
        ),
        "collision_actual_forbidden_contact_count": (
            collision_actual_forbidden_contacts
        ),
        "collision_actual_forbidden_contact_rate": (
            collision_actual_forbidden_contacts / max(evaluated_actions, 1)
        ),
        "collision_hard_violation_count": collision_hard_violations,
        "collision_hard_violation_rate": (
            collision_hard_violations / max(evaluated_actions, 1)
        ),
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
