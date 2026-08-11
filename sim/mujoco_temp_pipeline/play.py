"""Load a checkpoint and replay its deterministic policy in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from sim.mujoco.tasks.cube_lift import CubeLiftEnv
from sim.mujoco.tasks.cube_lift.config import (
    CubeLiftTaskConfig,
    is_cube_lift_success_reason,
)
from sim.mujoco.tasks.cube_lift.learning import (
    build_policy_action_transform,
    encode_policy_observation,
)

from .checkpoint import build_policy_from_checkpoint, check_task_config_compatibility
from .policy import resolve_device
from .train import DEFAULT_CHECKPOINT, DEFAULT_TASK_CONFIG


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a temporary PPO checkpoint in the Cube-Lift viewer."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until viewer closes.")
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs until viewer closes.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument(
        "--allow-task-config-mismatch",
        action="store_true",
        help="Diagnostics only: run a legacy checkpoint in the current environment.",
    )
    parser.add_argument(
        "--print-safety-events",
        action="store_true",
        help="Print projected/rejected collision-safety decisions during replay.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 0 or args.duration < 0.0:
        raise ValueError("episodes and duration must be non-negative.")
    device = resolve_device(args.device)
    policy, normalizer, payload = build_policy_from_checkpoint(
        args.checkpoint, device=device
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
    if compatibility["task_config_mismatch"]:
        print(
            "warning: legacy checkpoint/task mismatch enabled; "
            "use this rollout only for collision diagnostics.",
            flush=True,
        )

    from sim.mujoco.diagnostics.cube_drop import _load_mujoco_gui, _require_interactive_display
    from sim.mujoco.viewer import passive_viewer_handle

    _require_interactive_display()
    # Keep the existing display/GLFW preflight, but use the shared viewer
    # context below so the daemon GUI thread is drained before env/model
    # teardown. Direct ``viewer.launch_passive`` can segfault at interpreter
    # shutdown on Linux after a successful replay.
    _load_mujoco_gui()
    env = CubeLiftEnv(args.task_config)
    episodes = 0
    successes = 0
    policy_steps = 0
    total_raw_reward = 0.0
    projected_actions = 0
    rejected_actions = 0
    minimum_safety_distance = float("inf")
    hard_penetrations = 0
    tracking_error_sum = 0.0
    tracking_error_max = 0.0
    gripper_tracking_error_sum = 0.0
    gripper_tracking_error_max = 0.0
    last_safety_signature: tuple[object, ...] | None = None
    started = time.monotonic()
    try:
        observation, _info = env.reset(seed=args.seed)
        with passive_viewer_handle(
            env.scene,
            show_left_ui=not args.hide_left_ui,
            show_right_ui=not args.hide_right_ui,
        ) as handle:
            handle.cam.lookat[:] = (-0.472, 0.0, 0.15)
            handle.cam.distance = 1.05
            handle.cam.azimuth = 135.0
            handle.cam.elevation = -25.0
            while handle.is_running():
                frame_started = time.monotonic()
                if args.duration > 0.0 and frame_started - started >= args.duration:
                    break
                actor_observation = encode_policy_observation(
                    task_config,
                    observation,
                )
                with torch.no_grad():
                    action, _log_prob = policy.predict_action(
                        torch.as_tensor(
                            normalizer.normalize(actor_observation), device=device
                        ).unsqueeze(0),
                        deterministic=True,
                    )
                    environment_action = action_transform.transform(
                        action.cpu().numpy().astype(np.float32, copy=False),
                        observation=np.asarray(observation)[None, :],
                    )
                observation, reward, terminated, truncated, info = env.step(
                    environment_action.squeeze(0)
                )
                total_raw_reward += reward
                policy_steps += 1
                safety = info.get("safety") or {
                    "status": info.get("safety_status"),
                    "blocking_body1": info.get("safety_blocking_body1"),
                    "blocking_body2": info.get("safety_blocking_body2"),
                    "blocking_minimum_distance_m": info.get(
                        "safety_blocking_distance_m"
                    ),
                }
                modified = bool(
                    safety.get("modified", info.get("safety_modified", False))
                )
                alpha = float(
                    safety.get(
                        "projection_alpha",
                        info.get("safety_projection_alpha", 1.0),
                    )
                )
                projected_actions += int(modified)
                rejected_actions += int(alpha <= 1e-8)
                distance_value = safety.get(
                    "minimum_distance_m",
                    info.get("safety_minimum_distance_m"),
                )
                if distance_value is not None:
                    distance = float(distance_value)
                    if np.isfinite(distance):
                        minimum_safety_distance = min(
                            minimum_safety_distance, distance
                        )
                hard_penetrations += int(
                    info.get("termination_reason")
                    == "robot_environment_penetration"
                )
                tracking_error_sum += float(
                    info.get("arm_tracking_error_rad_mean", 0.0)
                )
                tracking_error_max = max(
                    tracking_error_max,
                    float(info.get("arm_tracking_error_rad_max", 0.0)),
                )
                gripper_error = abs(
                    float(info.get("right_gripper_tracking_error", 0.0))
                )
                gripper_tracking_error_sum += gripper_error
                gripper_tracking_error_max = max(
                    gripper_tracking_error_max, gripper_error
                )
                if getattr(args, "print_safety_events", False) and modified:
                    signature = (
                        safety.get("status"),
                        safety.get("blocking_body1"),
                        safety.get("blocking_body2"),
                    )
                    if signature != last_safety_signature or policy_steps % 15 == 0:
                        blocking_distance = safety.get("blocking_minimum_distance_m")
                        blocking_pair = (
                            f"{safety.get('blocking_body1', '?')}<->"
                            f"{safety.get('blocking_body2', '?')}"
                        )
                        print(
                            "[collision-safety] "
                            f"step={policy_steps} status={safety.get('status')} "
                            f"alpha={alpha:.6f} accepted_min_m={distance_value} "
                            f"blocking={blocking_pair} "
                            f"blocking_distance_m={blocking_distance}",
                            flush=True,
                        )
                    last_safety_signature = signature
                handle.sync()
                if terminated or truncated:
                    episodes += 1
                    successes += int(
                        is_cube_lift_success_reason(
                            info.get("termination_reason")
                        )
                    )
                    if args.episodes > 0 and episodes >= args.episodes:
                        break
                    observation, _info = env.reset(seed=args.seed + episodes)
                remaining = env.config.policy_period_sec - (time.monotonic() - frame_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()
    return {
        **compatibility,
        "checkpoint": str(args.checkpoint),
        "checkpoint_global_step": int(payload["global_step"]),
        "episodes": episodes,
        "successes": successes,
        "policy_steps": policy_steps,
        "total_raw_reward": total_raw_reward,
        "projected_action_count": projected_actions,
        "projected_action_rate": projected_actions / max(policy_steps, 1),
        "fully_rejected_action_count": rejected_actions,
        "fully_rejected_action_rate": rejected_actions / max(policy_steps, 1),
        "minimum_robot_environment_distance_m": (
            None
            if not np.isfinite(minimum_safety_distance)
            else minimum_safety_distance
        ),
        "hard_penetration_count": hard_penetrations,
        "arm_tracking_error_rad_mean": tracking_error_sum / max(policy_steps, 1),
        "arm_tracking_error_rad_max": tracking_error_max,
        "gripper_tracking_error_mean": (
            gripper_tracking_error_sum / max(policy_steps, 1)
        ),
        "gripper_tracking_error_max": gripper_tracking_error_max,
        "device": str(device),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = run(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
