"""CLI for the temporary batched Cube-Lift PPO smoke training."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .checkpoint import (
    build_warm_start_exploration_schedule,
    build_policy_from_checkpoint,
    check_task_config_compatibility,
    expand_cube_pose_policy_state,
    save_checkpoint,
    task_config_sha256,
    transfer_actor_state,
    transfer_policy_action_components,
    scale_pose_action_head_for_doubled_step,
)
from .bootstrap import (
    collect_goal_delta_bootstrap_dataset,
    fit_goal_delta_actor_bootstrap,
)
from .config import (
    GoalDeltaActorBootstrapConfig,
    PPOConfig,
    RewardShapingConfig,
)
from .normalization import RunningMeanStd
from .ppo import PPOSmokeTrainer
from .records import RunRecorder
from sim.mujoco.tasks.cube_lift.config import (
    CubeLiftTaskConfig,
    SCRIPTED_CONTACT_LOST_REASON,
)
from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata


# New training defaults to the state-only, no-episode-clock physical-contact
# profile. RGB profiles remain explicit compatibility options for old runs.
DEFAULT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json"
)
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "latest.pt"
DEFAULT_RECORD_ROOT = Path(__file__).resolve().parent / "records"


def initialize_policy_weights(
    trainer: PPOSmokeTrainer,
    checkpoint: str | Path,
    *,
    task_config_path: str | Path,
) -> dict[str, object]:
    """Warm-start policy/normalizer while resetting PPO optimizer and counters."""

    restored, normalizer, payload = build_policy_from_checkpoint(
        checkpoint,
        device=trainer.device,
    )
    critic_topology_changed = (
        (
            restored.critic_observation_dim,
            restored.separate_critic_backbone,
        )
        != (
            trainer.policy.critic_observation_dim,
            trainer.policy.separate_critic_backbone,
        )
    )
    compatibility = check_task_config_compatibility(
        payload,
        task_config_path,
        allow_mismatch=True,
        allow_cube_pose_observation_expansion=True,
        allow_policy_action_component_transfer=True,
        allow_policy_action_transform_transfer=True,
        allow_task_action_conditioning_transfer=True,
    )
    action_component_transfer = compatibility.get(
        "policy_action_component_transfer"
    )
    if (
        restored.environment_action_dim != trainer.policy.environment_action_dim
        or restored.active_action_indices
        != trainer.policy.active_action_indices
    ) and not isinstance(action_component_transfer, Mapping):
        raise ValueError(
            "Warm-start checkpoint policy action mapping is incompatible: "
            f"checkpoint={restored.active_action_indices}/"
            f"{restored.environment_action_dim}, "
            f"runtime={trainer.policy.active_action_indices}/"
            f"{trainer.policy.environment_action_dim}. Start a fresh checkpoint."
        )
    if bool(compatibility.get("cube_pose_observation_expansion", False)) and isinstance(
        action_component_transfer, Mapping
    ):
        raise ValueError(
            "A warm start cannot expand observations and remap action "
            "components in the same transfer. Use one explicit stage at a time."
        )
    if isinstance(action_component_transfer, Mapping):
        shared = transfer_policy_action_components(
            restored,
            trainer.policy,
            source_components=tuple(
                str(value)
                for value in action_component_transfer[
                    "checkpoint_components"
                ]
            ),
            target_components=tuple(
                str(value)
                for value in action_component_transfer["runtime_components"]
            ),
        )
        if normalizer.mean.shape != (trainer.observation_dim,):
            raise ValueError(
                "Action component transfer requires an unchanged observation "
                f"shape; checkpoint={normalizer.mean.shape}, "
                f"runtime={(trainer.observation_dim,)}."
            )
        trainer.observation_normalizer = normalizer
        action_component_transfer = {
            **dict(action_component_transfer),
            "transferred_components": shared,
        }
        compatibility["policy_action_component_transfer"] = (
            action_component_transfer
        )
    elif bool(compatibility.get("cube_pose_observation_expansion", False)):
        # Preserve the learned 28-D actor while adding the seven new cube-pose
        # inputs.  The new first-layer columns are zeroed, and the arm output
        # head is made conservative because v5 doubles the physical pose step.
        expand_cube_pose_policy_state(restored, trainer.policy)
        scale_pose_action_head_for_doubled_step(trainer.policy)
        if normalizer.mean.shape != (28,) or trainer.observation_dim != 35:
            raise ValueError(
                "Cube-pose migration expects a 28-D source and 35-D runtime "
                f"observation, got source={normalizer.mean.shape}, "
                f"runtime={trainer.observation_dim}."
            )
        expanded_normalizer = RunningMeanStd.create((35,), epsilon=normalizer.count)
        expanded_normalizer.mean[:28] = normalizer.mean
        expanded_normalizer.var[:28] = normalizer.var
        # Keep the new channels numerically quiet until the running statistics
        # have observed them in the target environment.  Their initial policy
        # weights are zero, so this does not alter the old actor's behavior.
        expanded_normalizer.mean[28:] = np.mean(trainer.observation[:, 28:35], axis=0)
        expanded_normalizer.var[28:] = 1.0
        expanded_normalizer.clip = normalizer.clip
        trainer.observation_normalizer = expanded_normalizer
    else:
        if critic_topology_changed:
            transfer_actor_state(restored, trainer.policy)
        else:
            trainer.policy.load_state_dict(restored.state_dict())
        trainer.observation_normalizer = normalizer
    exploration_schedule: dict[str, object] | None = None
    checkpoint_metadata = payload.get("task_metadata")
    checkpoint_learning = (
        checkpoint_metadata.get("learning_contract")
        if isinstance(checkpoint_metadata, Mapping)
        else None
    )
    source_components = (
        tuple(
            str(value)
            for value in checkpoint_learning.get(
                "policy_action_components",
                (),
            )
        )
        if isinstance(checkpoint_learning, Mapping)
        else ()
    )
    target_components = tuple(
        trainer.task_config.resolved_policy_action_components
    )
    if (
        len(source_components) == restored.action_dim
        and len(target_components) == trainer.action_dim
        and any(component in source_components for component in target_components)
    ):
        initial_log_std, final_log_std = (
            build_warm_start_exploration_schedule(
                source_log_std=(
                    restored.log_std.detach().cpu().numpy()
                ),
                source_components=source_components,
                target_components=target_components,
                configured_initial_log_std=(
                    trainer.config.initial_log_std
                ),
                configured_final_log_std=trainer.config.final_log_std,
                new_action_log_std_cap=(
                    trainer.config.warm_start_new_action_log_std_cap
                ),
            )
        )
        exploration_schedule = trainer.configure_exploration_schedule(
            initial_log_std,
            final_log_std,
            source="checkpoint_named_components",
        )
    if bool(compatibility.get("task_config_mismatch", False)):
        # Keep the actor as a warm start, but do not reuse value estimates
        # from a different transition/reward contract.
        trainer.policy.reset_critic()
    elif (
        trainer.critic_observation_normalizer is not None
        and not critic_topology_changed
    ):
        critic_normalizer_state = payload.get(
            "critic_observation_normalizer"
        )
        if not isinstance(critic_normalizer_state, Mapping):
            raise ValueError(
                "Privileged-critic warm start is missing its independent "
                "critic observation normalizer."
            )
        critic_normalizer = RunningMeanStd.from_state(
            critic_normalizer_state
        )
        if critic_normalizer.mean.shape != (
            trainer.critic_observation_dim,
        ):
            raise ValueError(
                "Critic observation normalizer shape does not match runtime: "
                f"checkpoint={critic_normalizer.mean.shape}, "
                f"runtime={(trainer.critic_observation_dim,)}."
            )
        trainer.critic_observation_normalizer = critic_normalizer
    return {
        "path": str(Path(checkpoint).resolve()),
        "source_global_step": int(payload.get("global_step", 0)),
        "critic_topology_changed": critic_topology_changed,
        "exploration_schedule": exploration_schedule,
        **compatibility,
    }


def _format_duration(seconds: float | None) -> str:
    """Format a duration for the human terminal summary."""

    if seconds is None or not math.isfinite(float(seconds)):
        return "--:--:--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


def _format_metric(
    value: Any,
    *,
    decimals: int = 3,
    suffix: str = "",
    fallback: str = "-",
    scientific: bool = False,
) -> str:
    if value is None:
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(number):
        return fallback
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if scientific:
        return f"{number:.{decimals}e}{suffix}"
    return f"{number:.{decimals}f}{suffix}"


def format_training_progress(
    record: Mapping[str, Any],
    *,
    total_timesteps: int,
    next_checkpoint: int | None = None,
    checkpoint_saved: bool = False,
) -> str:
    """Return the compact, human-readable line printed during training.

    The complete record is still persisted by :class:`RunRecorder`; this line
    deliberately contains only progress and the metrics needed while watching
    a run.  Keeping this formatter pure also makes the terminal contract easy
    to test without starting MuJoCo workers.
    """

    total = max(int(total_timesteps), 1)
    step = max(int(record.get("global_step", 0)), 0)
    progress = min(100.0, 100.0 * step / total)
    elapsed = float(record.get("wall_time_seconds", 0.0) or 0.0)
    tps = float(record.get("transitions_per_second", 0.0) or 0.0)
    remaining = max(total - step, 0)
    eta = remaining / tps if tps > 1e-9 else None
    success_rate = 100.0 * float(record.get("success_rate", 0.0) or 0.0)
    checkpoint = "saved" if checkpoint_saved else (
        f"next={int(next_checkpoint):,}" if next_checkpoint is not None else "final"
    )
    rgb_health = ""
    if record.get("wrist_rgb_health_rate") is not None:
        rgb_health = (
            f" rgbstd={_format_metric(record.get('wrist_rgb_std'), decimals=1)}"
            f" rgbok={_format_metric(100.0 * float(record.get('wrist_rgb_health_rate', 0.0) or 0.0), decimals=1, suffix='%')}"
        )
    action_health = ""
    if record.get("action_conditioning_scale_mean") is not None:
        action_health = (
            " act="
            f"{_format_metric(record.get('policy_action_abs_mean'))}"
            " sat="
            f"{_format_metric(100.0 * float(record.get('policy_action_saturation_rate', 0.0) or 0.0), decimals=1, suffix='%')}"
            " ascale="
            f"{_format_metric(record.get('action_conditioning_scale_mean'))}"
            " "
        )
    return (
        f"[train] steps={step:,}/{total:,} ({progress:5.1f}%) "
        f"u={int(record.get('update', 0)):04d} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)} "
        f"tps={tps:,.1f} "
        f"ep={int(record.get('completed_episodes', 0))} "
        f"succ={int(record.get('successful_episodes', 0))}({success_rate:4.1f}%) "
        f"rew={_format_metric(record.get('raw_reward_mean'), decimals=4)} "
        f"shape={_format_metric(record.get('shaped_reward_mean'), decimals=4)} "
        f"return={_format_metric(record.get('recent_episode_raw_return_mean'), decimals=3)} "
        f"reach={_format_metric(record.get('reward_distance_m_mean'), suffix='m')} "
        f"close={_format_metric(record.get('reward_gripper_close_mean'))} "
        f"contact={_format_metric(100.0 * float(record.get('reward_bilateral_contact_mean', 0.0) or 0.0), decimals=1, suffix='%')} "
        f"lift={_format_metric(record.get('reward_lift_fraction_mean'))} "
        f"{action_health}"
        f"proj={_format_metric(100.0 * float(record.get('safety_projection_rate', 0.0) or 0.0), decimals=1, suffix='%')} "
        f"reject={_format_metric(100.0 * float(record.get('safety_rejection_rate', 0.0) or 0.0), decimals=1, suffix='%')} "
        f"clr={int(record.get('clearance_violation_count', 0))} "
        f"pen={int(record.get('hard_penetration_count', 0))} "
        f"closs={int(record.get('contact_loss_termination_count', 0))} "
        f"jlim={int(record.get('joint_limit_termination_count', 0))} "
        f"tout={int(record.get('time_limit_termination_count', 0))} "
        f"armerr={_format_metric(record.get('arm_tracking_error_rad_mean'), decimals=4)}rad "
        f"griperr={_format_metric(record.get('gripper_tracking_error_mean'), decimals=3)} "
        f"{rgb_health}"
        f"std={_format_metric(record.get('policy_std_mean'))} "
        f"loss={_format_metric(record.get('policy_loss'), decimals=3)} "
        f"v={_format_metric(record.get('value_loss'), decimals=3)} "
        f"vclip={_format_metric(100.0 * float(record.get('value_clip_fraction', 0.0) or 0.0), decimals=1, suffix='%')} "
        f"ev={_format_metric(record.get('explained_variance'), decimals=3)} "
        f"grad={_format_metric(record.get('actor_grad_norm'), decimals=2)}/"
        f"{_format_metric(record.get('critic_grad_norm'), decimals=2)} "
        f"kl={_format_metric(record.get('approx_kl'), decimals=2, scientific=True)} "
        f"stop={_format_metric(100.0 * float(record.get('ppo_early_stop_rate', 0.0) or 0.0), decimals=1, suffix='%')} "
        f"ckpt={checkpoint}"
    )


def format_training_start(
    *,
    task_name: str,
    policy_class: str,
    observation_dim: int,
    policy_action_dim: int,
    environment_action_dim: int,
    total_timesteps: int,
    num_envs: int,
    rollout_steps: int,
    device: str,
    seed: int,
    reward_shaping: bool,
    checkpoint: Path,
    record_directory: Path,
) -> str:
    return (
        f"[train] start task={task_name} policy={policy_class} "
        f"obs={observation_dim} "
        f"action={policy_action_dim}->{environment_action_dim} "
        f"envs={num_envs} rollout={rollout_steps} "
        f"total={int(total_timesteps):,} device={device} seed={seed} "
        f"reward={'shaped' if reward_shaping else 'raw'} "
        f"checkpoint={checkpoint} records={record_directory}"
    )


def format_training_complete(summary: Mapping[str, Any]) -> str:
    return (
        f"[train] done steps={int(summary.get('global_step', 0)):,} "
        f"updates={int(summary.get('updates', 0))} "
        f"elapsed={_format_duration(summary.get('elapsed_seconds'))} "
        f"tps={_format_metric(summary.get('transitions_per_second'), decimals=1)} "
        f"episodes={int(summary.get('completed_episodes', 0))} "
        f"success={int(summary.get('successful_episodes', 0))} "
        f"rate={100.0 * float(summary.get('success_rate', 0.0) or 0.0):.1f}% "
        f"device={summary.get('device', '-')} "
        f"checkpoint={summary.get('checkpoint', '-')} "
        f"records={summary.get('record_directory', '-')}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a minimal PPO directly on CubeLiftVectorExecutor."
    )
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start compatible policy weights and observation normalization "
            "without restoring optimizer state or training counters."
        ),
    )
    parser.add_argument(
        "--goal-delta-bootstrap-transitions",
        type=int,
        default=0,
        help=(
            "Optionally initialize the xyz actor from its deployable "
            "reach-target-delta feature before PPO. Zero (default) keeps "
            "random PPO initialization."
        ),
    )
    parser.add_argument(
        "--goal-delta-bootstrap-epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--goal-delta-bootstrap-minibatch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--goal-delta-bootstrap-learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--goal-delta-bootstrap-target-mse",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--goal-delta-bootstrap-lift-step-m",
        type=float,
        default=0.004,
        help=(
            "Actor-visible Pick teacher z increment after measured closure; "
            "ignored by Reach and Contact bootstrap profiles."
        ),
    )
    parser.add_argument(
        "--goal-delta-bootstrap-pick-close-distance-m",
        type=float,
        default=0.035,
        help="Pick teacher distance for beginning and then maintaining closure.",
    )
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--goal-delta-bootstrap-lock-lift-xy",
        action="store_true",
        help="Set Pick teacher XY requests to zero in lift, without adding actor phase input.",
    )
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="Stop PPO minibatch updates when approximate KL exceeds this value.",
    )
    parser.add_argument(
        "--max-log-ratio",
        type=float,
        default=10.0,
        help="Clamp PPO log importance ratios before exponentiation.",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=10.0,
        help="Bound normalized advantages for numerical robustness.",
    )
    parser.add_argument(
        "--value-clip-coef",
        type=float,
        default=0.0,
        help=(
            "Optionally clip value updates around rollout predictions; zero "
            "preserves the validated shared-backbone baseline."
        ),
    )
    parser.add_argument(
        "--actor-reference-mse-coef",
        type=float,
        default=0.0,
        help=(
            "Optionally anchor PPO's deterministic actor output to the "
            "post-initialization actor using deployable observations only."
        ),
    )
    parser.add_argument("--initial-log-std", type=float, default=-0.5)
    parser.add_argument("--final-log-std", type=float, default=-2.5)
    parser.add_argument("--std-anneal-fraction", type=float, default=0.75)
    parser.add_argument(
        "--warm-start-new-action-log-std-cap",
        type=float,
        default=-1.5,
        help=(
            "Maximum initial log std for action components added by a named "
            "checkpoint transfer; shared components retain checkpoint std."
        ),
    )
    parser.add_argument(
        "--effective-action-for-ppo",
        action="store_true",
        help=(
            "Deprecated and rejected: projected/executed actions cannot "
            "replace PPO's sampled latent action."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--critic-observation-mode",
        choices=("actor", "privileged_training_state"),
        default="actor",
        help=(
            "Use the deployable actor observation for the critic (default), "
            "or opt in to a separate simulator-only 31-D training critic."
        ),
    )
    parser.add_argument(
        "--critic-backbone-mode",
        choices=("separate", "shared_actor"),
        default="shared_actor",
        help=(
            "Share the actor representation for same-observation training "
            "(default), or isolate value gradients with an independent "
            "critic encoder. Privileged critic input always selects the "
            "independent encoder."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--start-method", default="spawn", choices=("spawn", "forkserver"))
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--log-interval-updates",
        type=int,
        default=1,
        help=(
            "Print one compact progress line every N PPO updates; complete "
            "metrics are always written to metrics.jsonl/csv."
        ),
    )
    parser.add_argument(
        "--print-full-metrics",
        action="store_true",
        help="Print the complete per-update JSON records instead of compact lines.",
    )
    parser.add_argument(
        "--no-reward-shaping",
        action="store_true",
        help="Train on the original sparse Cube-Lift reward only.",
    )
    parser.add_argument(
        "--require-rgb-health",
        action="store_true",
        help=(
            "Abort immediately when the reset wrist RGB frame is flat/dark/"
            "saturated; diagnostics are still recorded without this flag."
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if int(args.log_interval_updates) <= 0:
        raise ValueError("log_interval_updates must be positive.")
    ppo_config = PPOConfig(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        max_log_ratio=args.max_log_ratio,
        advantage_clip=args.advantage_clip,
        value_clip_coef=args.value_clip_coef,
        actor_reference_mse_coef=args.actor_reference_mse_coef,
        initial_log_std=args.initial_log_std,
        final_log_std=args.final_log_std,
        std_anneal_fraction=args.std_anneal_fraction,
        warm_start_new_action_log_std_cap=(
            args.warm_start_new_action_log_std_cap
        ),
        effective_action_for_ppo=args.effective_action_for_ppo,
        hidden_dim=args.hidden_dim,
        critic_observation_mode=args.critic_observation_mode,
        critic_backbone_mode=args.critic_backbone_mode,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
        start_method=args.start_method,
        device=args.device,
    )
    bootstrap_config = GoalDeltaActorBootstrapConfig(
        transitions=args.goal_delta_bootstrap_transitions,
        epochs=args.goal_delta_bootstrap_epochs,
        minibatch_size=args.goal_delta_bootstrap_minibatch_size,
        learning_rate=args.goal_delta_bootstrap_learning_rate,
        target_mse=args.goal_delta_bootstrap_target_mse,
        lift_step_m=args.goal_delta_bootstrap_lift_step_m,
        pick_lift_lock_xy=args.goal_delta_bootstrap_lock_lift_xy,
        pick_close_distance_m=(
            args.goal_delta_bootstrap_pick_close_distance_m
        ),
    )
    task_config = CubeLiftTaskConfig.load(args.task_config)
    reward_values = dict(task_config.reward_shaping)
    reward_values["enabled"] = not args.no_reward_shaping
    reward_config = RewardShapingConfig.from_dict(reward_values)
    model_cache_dir = (
        None
        if args.model_cache_dir is None
        else str(args.model_cache_dir)
    )
    bootstrap_dataset = (
        collect_goal_delta_bootstrap_dataset(
            bootstrap_config,
            task_config_path=str(args.task_config),
            num_envs=ppo_config.num_envs,
            seed=ppo_config.seed,
            start_method=ppo_config.start_method,
            model_cache_dir=model_cache_dir,
        )
        if bootstrap_config.enabled
        else None
    )
    trainer = PPOSmokeTrainer(
        ppo_config,
        reward_config,
        task_config_path=str(args.task_config),
        model_cache_dir=model_cache_dir,
        require_wrist_rgb_health=bool(args.require_rgb_health),
    )
    try:
        checkpoint_initialization = (
            None
            if args.init_checkpoint is None
            else initialize_policy_weights(
                trainer,
                args.init_checkpoint,
                task_config_path=args.task_config,
            )
        )
        if bootstrap_dataset is not None:
            (
                trainer.observation_normalizer,
                bootstrap_initialization,
            ) = fit_goal_delta_actor_bootstrap(
                trainer.policy,
                bootstrap_dataset,
                bootstrap_config,
                device=trainer.device,
            )
            if checkpoint_initialization is None:
                policy_initialization: dict[str, object] = {
                    "type": "goal_delta_actor_bootstrap",
                    **bootstrap_initialization,
                }
            else:
                policy_initialization = {
                    "type": (
                        "checkpoint_warm_start_then_goal_delta_actor_bootstrap"
                    ),
                    "checkpoint_warm_start": checkpoint_initialization,
                    "goal_delta_actor_bootstrap": bootstrap_initialization,
                }
        elif checkpoint_initialization is not None:
            policy_initialization = {
                "type": "checkpoint_warm_start",
                **checkpoint_initialization,
            }
        else:
            policy_initialization = {"type": "random"}
        if float(ppo_config.actor_reference_mse_coef) > 0.0:
            if policy_initialization["type"] == "random":
                raise ValueError(
                    "actor_reference_mse_coef requires an initialization "
                    "checkpoint or goal-delta bootstrap."
                )
            actor_reference = trainer.capture_actor_reference(
                source=str(policy_initialization["type"]),
            )
        else:
            actor_reference = {
                "enabled": False,
                "source": None,
                "mse_coefficient": 0.0,
                "uses_privileged_observation": False,
            }
    except BaseException:
        # Initialization failures happen before the main training finally
        # block. Close vector workers here so a bad bootstrap/checkpoint cannot
        # strand process-local MuJoCo models in host memory.
        trainer.close()
        raise
    learning_metadata = build_learning_metadata(trainer.task_config)
    recorder = RunRecorder(
        args.record_root,
        run_name=args.run_name,
        config={
            "schema_version": "dexteleop.sim.mujoco_temp_ppo_run.v1",
            "ppo": ppo_config.to_dict(),
            "goal_delta_actor_bootstrap": bootstrap_config.to_dict(),
            "reward_shaping": reward_config.to_dict(),
            "task_config": str(Path(args.task_config).resolve()),
            "task_config_sha256": task_config_sha256(args.task_config),
            "task_metadata": trainer.task_config.to_metadata(),
            "learning_contract": learning_metadata,
            "policy_action_mapping": {
                "policy_action_dim": trainer.action_dim,
                "environment_action_dim": trainer.environment_action_dim,
                "active_action_indices": list(
                    trainer.active_action_indices
                ),
                "policy_action_components": list(
                    trainer.policy_action_transform.policy_components
                ),
                "task_action_components": list(
                    trainer.policy_action_transform.task_components
                ),
                "transform_id": trainer.policy_action_transform.transform_id,
                "transform_hash": (
                    trainer.policy_action_transform.semantic_content_hash
                ),
            },
            "exploration_schedule": trainer.exploration_schedule_metadata,
            "actor_reference": actor_reference,
            "observation_ownership": {
                "task_observation_dim": trainer.task_observation_dim,
                "actor_observation_dim": trainer.observation_dim,
                "actor_observation_fields": list(
                    trainer.learning_model_input_binding.input_fields
                ),
                "actor_uses_privileged": bool(
                    learning_metadata["actor_uses_privileged"]
                ),
                "critic_observation_mode": trainer.critic_observation_mode,
                "critic_observation_dim": trainer.critic_observation_dim,
                "critic_backbone_mode": trainer.critic_backbone_mode,
            },
            "checkpoint_target": str(Path(args.checkpoint).resolve()),
            "initialization_checkpoint": checkpoint_initialization,
            "policy_initialization": policy_initialization,
            "model_cache_dir": (
                None if args.model_cache_dir is None else str(Path(args.model_cache_dir).resolve())
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "resolved_device": str(trainer.device),
            "pose_ik_solvers": list(trainer.pose_ik_solvers),
            "log_interval_updates": int(args.log_interval_updates),
            "print_full_metrics": bool(args.print_full_metrics),
            "require_rgb_health": bool(args.require_rgb_health),
        },
    )
    if args.print_full_metrics:
        print(
            json.dumps(
                {
                    "event": "train_start",
                    "task": trainer.task_config.name,
                    "policy_class": trainer.task_config.policy_class,
                    "observation_dim": trainer.observation_dim,
                    "task_observation_dim": trainer.task_observation_dim,
                    "critic_observation_mode": trainer.critic_observation_mode,
                    "critic_observation_dim": trainer.critic_observation_dim,
                    "critic_backbone_mode": trainer.critic_backbone_mode,
                    "policy_action_dim": trainer.action_dim,
                    "environment_action_dim": trainer.environment_action_dim,
                    "active_action_indices": list(
                        trainer.active_action_indices
                    ),
                    "total_timesteps": ppo_config.total_timesteps,
                    "num_envs": ppo_config.num_envs,
                    "rollout_steps": ppo_config.rollout_steps,
                    "device": str(trainer.device),
                    "seed": ppo_config.seed,
                    "reward_shaping": reward_config.enabled,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "record_directory": str(recorder.directory),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            format_training_start(
                task_name=trainer.task_config.name,
                policy_class=trainer.task_config.policy_class,
                observation_dim=trainer.observation_dim,
                policy_action_dim=trainer.action_dim,
                environment_action_dim=trainer.environment_action_dim,
                total_timesteps=ppo_config.total_timesteps,
                num_envs=ppo_config.num_envs,
                rollout_steps=ppo_config.rollout_steps,
                device=str(trainer.device),
                seed=ppo_config.seed,
                reward_shaping=reward_config.enabled,
                checkpoint=Path(args.checkpoint).resolve(),
                record_directory=recorder.directory,
            ),
            flush=True,
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
            rollout_termination_reasons = [
                episode.get("termination_reason") for episode in rollout.episodes
            ]
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
                "policy_log_std_mean": float(
                    trainer.policy.log_std.detach().mean().cpu()
                ),
                "policy_log_std_by_component": (
                    trainer.policy.log_std.detach().cpu().numpy().astype(float).tolist()
                ),
                "policy_action_dim": trainer.action_dim,
                "environment_action_dim": trainer.environment_action_dim,
                **rollout.action_diagnostics,
                **rollout.pose_target_diagnostics,
                "learning_rate": float(trainer.optimizer.param_groups[0]["lr"]),
                "safety_projection_rate": rollout.safety_projection_rate,
                "safety_rejection_rate": rollout.safety_rejection_rate,
                "safety_recovery_rate": rollout.safety_recovery_rate,
                "safety_projection_alpha_mean": rollout.safety_projection_alpha_mean,
                **{
                    f"safety_collision_{category}_rate": rate
                    for category, rate in (
                        rollout.safety_collision_category_rates.items()
                    )
                },
                # Task-independent collision cost components remain separate
                # from shaped reward and actor inputs. A future constrained
                # optimizer can consume them without redefining task success.
                "collision_preventive_event_rate": (
                    rollout.collision_preventive_event_rate
                ),
                "collision_preventive_blocked_fraction_mean": (
                    rollout.collision_preventive_blocked_fraction_mean
                ),
                "collision_preventive_streak_steps_max": (
                    rollout.collision_preventive_streak_steps_max
                ),
                **{
                    f"collision_preventive_event_{category}_rate": rate
                    for category, rate in (
                        rollout.collision_preventive_event_category_rates.items()
                    )
                },
                "collision_actual_forbidden_contact_rate": (
                    rollout.collision_actual_forbidden_contact_rate
                ),
                "collision_hard_violation_rate": (
                    rollout.collision_hard_violation_rate
                ),
                "downstream_action_modified_rate": (
                    rollout.effective_action_override_rate
                ),
                "task_to_admitted_action_delta_mean": (
                    rollout.effective_action_delta_mean
                ),
                "effective_action_override_rate": (
                    rollout.effective_action_override_rate
                ),
                "effective_action_delta_mean": rollout.effective_action_delta_mean,
                "post_contact_phase_active_rate": (
                    rollout.post_contact_phase_active_rate
                ),
                "post_contact_action_modified_rate": (
                    rollout.post_contact_action_modified_rate
                ),
                "scripted_pregrasp_action_modified_rate": (
                    rollout.scripted_pregrasp_action_modified_rate
                ),
                "scripted_contact_action_modified_rate": (
                    rollout.scripted_contact_action_modified_rate
                ),
                "safety_minimum_distance_m": rollout.safety_minimum_distance_m,
                "hard_penetration_count": rollout.hard_penetration_count,
                "clearance_violation_count": rollout.clearance_violation_count,
                "contact_loss_termination_count": sum(
                    reason == SCRIPTED_CONTACT_LOST_REASON
                    for reason in rollout_termination_reasons
                ),
                "joint_limit_termination_count": sum(
                    reason == "sim_joint_position_limit_violation"
                    for reason in rollout_termination_reasons
                ),
                "time_limit_termination_count": sum(
                    reason == "time_limit"
                    for reason in rollout_termination_reasons
                ),
                "arm_tracking_error_rad_mean": rollout.arm_tracking_error_rad_mean,
                "arm_tracking_error_rad_max": rollout.arm_tracking_error_rad_max,
                "gripper_tracking_error_mean": rollout.gripper_tracking_error_mean,
                "gripper_tracking_error_max": rollout.gripper_tracking_error_max,
                "wrist_rgb_mean": rollout.wrist_rgb_mean,
                "wrist_rgb_std": rollout.wrist_rgb_std,
                "wrist_rgb_health_rate": rollout.wrist_rgb_health_rate,
                **rollout.reward_term_means,
                **losses,
            }
            recorder.record_update(record, rollout.episodes)
            checkpoint_saved = False
            if trainer.global_step >= next_checkpoint:
                save_checkpoint(
                    args.checkpoint,
                    policy=trainer.policy,
                    optimizer=trainer.optimizer,
                    observation_normalizer=trainer.observation_normalizer,
                    critic_observation_normalizer=(
                        trainer.critic_observation_normalizer
                    ),
                    ppo_config=ppo_config,
                    reward_config=reward_config,
                    global_step=trainer.global_step,
                    update=trainer.update_index,
                    task_config_path=args.task_config,
                    policy_initialization=policy_initialization,
                )
                save_checkpoint(
                    recorder.checkpoint_directory / f"step_{trainer.global_step:09d}.pt",
                    policy=trainer.policy,
                    optimizer=trainer.optimizer,
                    observation_normalizer=trainer.observation_normalizer,
                    critic_observation_normalizer=(
                        trainer.critic_observation_normalizer
                    ),
                    ppo_config=ppo_config,
                    reward_config=reward_config,
                    global_step=trainer.global_step,
                    update=trainer.update_index,
                    task_config_path=args.task_config,
                    policy_initialization=policy_initialization,
                )
                while next_checkpoint <= trainer.global_step:
                    next_checkpoint += ppo_config.checkpoint_interval
                checkpoint_saved = True
            should_print = (
                args.print_full_metrics
                or checkpoint_saved
                or trainer.update_index % int(args.log_interval_updates) == 0
                or trainer.global_step >= ppo_config.total_timesteps
            )
            if should_print:
                if args.print_full_metrics:
                    print(json.dumps(record, sort_keys=True), flush=True)
                else:
                    print(
                        format_training_progress(
                            record,
                            total_timesteps=ppo_config.total_timesteps,
                            next_checkpoint=next_checkpoint,
                            checkpoint_saved=checkpoint_saved,
                        ),
                        flush=True,
                    )

        destination = save_checkpoint(
            args.checkpoint,
            policy=trainer.policy,
            optimizer=trainer.optimizer,
            observation_normalizer=trainer.observation_normalizer,
            critic_observation_normalizer=(
                trainer.critic_observation_normalizer
            ),
            ppo_config=ppo_config,
            reward_config=reward_config,
            global_step=trainer.global_step,
            update=trainer.update_index,
            task_config_path=args.task_config,
            policy_initialization=policy_initialization,
        )
        run_checkpoint = save_checkpoint(
            recorder.checkpoint_directory / f"step_{trainer.global_step:09d}_final.pt",
            policy=trainer.policy,
            optimizer=trainer.optimizer,
            observation_normalizer=trainer.observation_normalizer,
            critic_observation_normalizer=(
                trainer.critic_observation_normalizer
            ),
            ppo_config=ppo_config,
            reward_config=reward_config,
            global_step=trainer.global_step,
            update=trainer.update_index,
            task_config_path=args.task_config,
            policy_initialization=policy_initialization,
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
            "task_observation_dim": trainer.task_observation_dim,
            "actor_observation_dim": trainer.observation_dim,
            "critic_observation_mode": trainer.critic_observation_mode,
            "critic_observation_dim": trainer.critic_observation_dim,
            "policy_initialization": policy_initialization,
            "actor_reference": actor_reference,
        }
        recorder.finish(summary)
        return summary
    finally:
        trainer.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run(args)
    if args.print_full_metrics:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_training_complete(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
