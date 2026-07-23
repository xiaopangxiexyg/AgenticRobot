"""Headless smoke runner for the task-local Cube-Lift Gymnasium environment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from sim.mujoco.tasks.cube_lift import (
    DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
    CubeLiftEnv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the headless CubeLiftEnv. This is a kinematic-direct-qpos "
            "debug task, not a physical-grasp or sim2real benchmark."
        )
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
        help="Task-local CubeLift JSON config.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Maximum policy steps. Default: 1 headless step; in --viewer mode, "
            "run until the window closes. 0 means unlimited in viewer mode."
        ),
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=7,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "GRIPPER"),
        default=(0.0,) * 7,
        help="One normalized 7-D action held on each requested policy step.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive MuJoCo GUI and drive this same CubeLiftEnv.",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Sample one bounded random 7-D right-arm/gripper action at each policy tick.",
    )
    parser.add_argument(
        "--random-action-scale",
        type=float,
        default=0.25,
        help="Scale of random normalized actions in [0, 1] (default: 0.25).",
    )
    parser.add_argument(
        "--viewer-duration",
        type=float,
        default=0.0,
        help="Viewer lifetime in wall-clock seconds; 0 keeps it open until closed.",
    )
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    steps = 1 if args.steps is None else int(args.steps)
    if steps < 0:
        raise ValueError("--steps must be non-negative.")
    action = np.asarray(args.action, dtype=np.float32).reshape(7)
    env = CubeLiftEnv(args.task_config)
    try:
        observation, info = env.reset(seed=args.seed)
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(steps):
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return {
            "task_config": str(args.task_config),
            "seed": int(args.seed),
            "policy_steps_requested": steps,
            "policy_steps_completed": int(info["episode_step"]),
            "total_reward": float(total_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "observation_dim": int(observation.size),
            "cube_position_m": info["cube_position_m"],
            "cube_z_m": info["cube_z_m"],
            "termination_reason": info["termination_reason"],
            "contacts": info["contacts"],
            "scheduler": info["scheduler"],
            "execution": {
                key: info[key]
                for key in ("execution_mode", "robot_dynamics", "grasp_mode")
            },
        }
    finally:
        env.close()


def run_viewer(args: argparse.Namespace) -> dict[str, object]:
    """Visualize random/fixed actions while stepping the same environment API."""

    if args.steps is not None and int(args.steps) < 0:
        raise ValueError("--steps must be non-negative.")
    if not 0.0 <= float(args.random_action_scale) <= 1.0:
        raise ValueError("--random-action-scale must lie in [0, 1].")
    if float(args.viewer_duration) < 0.0:
        raise ValueError("--viewer-duration must be non-negative.")

    # Reuse the existing Linux display/GLFW preflight used by the cube-drop GUI.
    from .cube_drop import _load_mujoco_gui, _require_interactive_display

    _require_interactive_display()
    _mujoco, viewer = _load_mujoco_gui()
    fixed_action = np.asarray(args.action, dtype=np.float32).reshape(7)
    rng = np.random.default_rng(int(args.seed))
    max_steps = 0 if args.steps is None else int(args.steps)
    duration_sec = float(args.viewer_duration)
    env = CubeLiftEnv(args.task_config)
    try:
        observation, info = env.reset(seed=int(args.seed))
        total_reward = 0.0
        completed_steps = 0
        resets = 0
        started_at = time.monotonic()
        with viewer.launch_passive(
            env.scene.model,
            env.scene.data,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        ) as handle:
            handle.cam.lookat[:] = (-0.472, 0.0, 0.15)
            handle.cam.distance = 1.05
            handle.cam.azimuth = 135.0
            handle.cam.elevation = -25.0
            while handle.is_running():
                frame_started_at = time.monotonic()
                if duration_sec > 0.0 and frame_started_at - started_at >= duration_sec:
                    break
                if max_steps > 0 and completed_steps >= max_steps:
                    break
                action = (
                    rng.uniform(
                        -float(args.random_action_scale),
                        float(args.random_action_scale),
                        size=7,
                    ).astype(np.float32)
                    if bool(args.random_actions)
                    else fixed_action
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                completed_steps += 1
                handle.sync()
                if terminated or truncated:
                    resets += 1
                    observation, info = env.reset(seed=int(args.seed) + resets)
                remaining = env.config.policy_period_sec - (time.monotonic() - frame_started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
        return {
            "viewer": True,
            "random_actions": bool(args.random_actions),
            "random_action_scale": float(args.random_action_scale),
            "policy_steps_completed": completed_steps,
            "resets": resets,
            "total_reward": float(total_reward),
            "observation_dim": int(observation.size),
            "cube_position_m": info["cube_position_m"],
            "cube_z_m": info["cube_z_m"],
            "execution": {
                key: info[key]
                for key in ("execution_mode", "robot_dynamics", "grasp_mode")
            },
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_viewer(args) if bool(args.viewer) else run(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
