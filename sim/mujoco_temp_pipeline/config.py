"""Small, explicit configuration objects for the temporary PPO pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from sim.mujoco.collision_safety import FORBIDDEN_COLLISION_CATEGORIES


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    # Legacy v0 keeps the historical narrow grasp gate.  The staged profile
    # used by normal_rgb_v3 exposes an earlier close gate while retaining the
    # strict gate for lift confidence and success.
    staged_grasp: bool = False
    reach_reward_scale: float = 0.25
    reach_sigma_m: float = 0.25
    reach_progress_scale: float = 4.0
    # Privileged-only guidance toward the scripted, collision-aware approach
    # direction. It is zero by default and never enters the policy contract.
    approach_alignment_reward_scale: float = 0.0
    grasp_height_offset_m: float = 0.03
    closed_grasp_height_offset_m: float = 0.025
    grasp_distance_m: float = 0.05
    grasp_gate_width_m: float = 0.01
    close_gate_distance_m: float = 0.12
    close_gate_width_m: float = 0.03
    near_reach_decay: float = 0.75
    grasp_region_reward_scale: float = 0.15
    # Potential-based progress toward the closed-grasp target.  Unlike the
    # absolute grasp-region term, this is zero when stationary, positive when
    # approaching the target, and negative when retreating from it.
    grasp_region_progress_scale: float = 0.0
    gripper_reward_scale: float = 0.25
    gripper_progress_scale: float = 2.0
    # Optional immediate credit for a positive gripper command.  The physical
    # actuator has several policy ticks of lag, so near-grasp curricula may
    # enable this without changing the deployable observation contract.
    gripper_command_reward_scale: float = 0.0
    premature_close_penalty: float = 0.02
    # Optional state penalty for remaining closed before a valid inner-surface
    # contact.  This is separate from the closing-command penalty: without it,
    # a policy can drift to the closed limit and keep that state forever.
    premature_close_state_penalty: float = 0.0
    grasp_acquire_close: float = 0.45
    grasp_acquire_gate: float = 0.25
    grasp_acquire_bonus: float = 1.0
    grasp_close_confidence_width: float = 0.05
    bilateral_contact_reward_scale: float = 0.0
    bilateral_contact_acquire_bonus: float = 0.0
    bilateral_contact_missing_penalty: float = 0.0
    # Contact geometry shaping is deliberately separate from the binary
    # bilateral-contact term: a top/bottom collision is not a grasp, even when
    # both fingers happen to touch the cube.
    side_contact_reward_scale: float = 0.0
    top_bottom_contact_penalty: float = 0.0
    # Generic gripper-surface contact shaping.  The scores come from a
    # surface contract (inner half-space versus outer half-space), not from a
    # particular object shape or object-local face label.
    inner_contact_reward_scale: float = 0.0
    bilateral_inner_contact_reward_scale: float = 0.0
    outer_contact_penalty: float = 0.0
    # New physical-grasp profiles can explicitly prevent closure/acquisition
    # shaping from being harvested before a valid inner-surface contact.
    gripper_reward_requires_contact: bool = False
    grasp_acquire_requires_contact: bool = False
    require_generic_contact_for_lift: bool = False
    contact_reward_onset_only: bool = False
    contact_missing_requires_previous: bool = False
    contact_gate_threshold: float = 0.5
    require_bilateral_contact_for_lift_reward: bool = False
    ungated_negative_lift_progress: bool = False
    # Penalize a closed gripper whose grasp-center frame enters the cube-center
    # exclusion radius before a valid bilateral inner-surface grasp exists.
    # A zero radius/scale keeps legacy profiles unchanged.
    center_proximity_limit_m: float = 0.0
    center_proximity_penalty: float = 0.0
    center_proximity_close_threshold: float = 0.35
    latch_reward_scale: float = 0.50
    lift_reward_scale: float = 6.0
    lift_progress_scale: float = 30.0
    success_bonus: float = 20.0
    action_penalty: float = 0.01
    safety_projection_penalty: float = 0.25
    # Fixed cost whenever collision admission modifies the requested action.
    # The existing alpha-weighted term measures intervention magnitude, but a
    # policy can otherwise harvest many tiny boundary projections almost for
    # free. An accepted monotonic recovery is deliberately not charged.
    safety_intervention_penalty: float = 0.0
    # Optional fixed costs selected by semantic forbidden-pair category.
    # These are reward-only signals from canonical admission and never actor
    # observations. Empty by default to preserve historical behavior.
    safety_collision_category_penalties: Mapping[str, float] = field(
        default_factory=dict
    )
    time_limit_penalty: float = 0.0
    failure_termination_penalty: float = 0.0
    unsafe_termination_penalty: float = 0.0

    def __post_init__(self) -> None:
        non_negative = {
            "reach_reward_scale": self.reach_reward_scale,
            "reach_progress_scale": self.reach_progress_scale,
            "approach_alignment_reward_scale": self.approach_alignment_reward_scale,
            "grasp_region_reward_scale": self.grasp_region_reward_scale,
            "grasp_region_progress_scale": self.grasp_region_progress_scale,
            "gripper_reward_scale": self.gripper_reward_scale,
            "gripper_progress_scale": self.gripper_progress_scale,
            "gripper_command_reward_scale": self.gripper_command_reward_scale,
            "latch_reward_scale": self.latch_reward_scale,
            "premature_close_penalty": self.premature_close_penalty,
            "premature_close_state_penalty": self.premature_close_state_penalty,
            "lift_reward_scale": self.lift_reward_scale,
            "lift_progress_scale": self.lift_progress_scale,
            "success_bonus": self.success_bonus,
            "action_penalty": self.action_penalty,
            "safety_projection_penalty": self.safety_projection_penalty,
            "safety_intervention_penalty": self.safety_intervention_penalty,
            "grasp_acquire_bonus": self.grasp_acquire_bonus,
            "bilateral_contact_reward_scale": self.bilateral_contact_reward_scale,
            "bilateral_contact_acquire_bonus": self.bilateral_contact_acquire_bonus,
            "bilateral_contact_missing_penalty": (
                self.bilateral_contact_missing_penalty
            ),
            "side_contact_reward_scale": self.side_contact_reward_scale,
            "top_bottom_contact_penalty": self.top_bottom_contact_penalty,
            "inner_contact_reward_scale": self.inner_contact_reward_scale,
            "bilateral_inner_contact_reward_scale": (
                self.bilateral_inner_contact_reward_scale
            ),
            "outer_contact_penalty": self.outer_contact_penalty,
            "time_limit_penalty": self.time_limit_penalty,
            "failure_termination_penalty": self.failure_termination_penalty,
            "unsafe_termination_penalty": self.unsafe_termination_penalty,
            "center_proximity_penalty": self.center_proximity_penalty,
        }
        invalid = {name: value for name, value in non_negative.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Reward scales must be non-negative: {invalid}.")
        category_penalties = {
            str(name): float(value)
            for name, value in self.safety_collision_category_penalties.items()
        }
        unknown_categories = sorted(
            set(category_penalties) - set(FORBIDDEN_COLLISION_CATEGORIES)
        )
        if unknown_categories:
            raise ValueError(
                "Unknown safety collision categories: "
                f"{unknown_categories}."
            )
        invalid_category_penalties = {
            name: value
            for name, value in category_penalties.items()
            if not math.isfinite(value) or value < 0.0
        }
        if invalid_category_penalties:
            raise ValueError(
                "Safety collision category penalties must be finite and "
                f"non-negative: {invalid_category_penalties}."
            )
        object.__setattr__(
            self,
            "safety_collision_category_penalties",
            dict(sorted(category_penalties.items())),
        )
        positive = {
            "reach_sigma_m": self.reach_sigma_m,
            "grasp_distance_m": self.grasp_distance_m,
            "grasp_gate_width_m": self.grasp_gate_width_m,
            "close_gate_distance_m": self.close_gate_distance_m,
            "close_gate_width_m": self.close_gate_width_m,
            "grasp_close_confidence_width": self.grasp_close_confidence_width,
            "grasp_height_offset_m": self.grasp_height_offset_m,
            "closed_grasp_height_offset_m": self.closed_grasp_height_offset_m,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0.0}
        if invalid:
            raise ValueError(f"Reward distances must be positive: {invalid}.")
        if self.closed_grasp_height_offset_m > self.grasp_height_offset_m:
            raise ValueError(
                "closed_grasp_height_offset_m cannot exceed grasp_height_offset_m."
            )
        unit_values = {
            "near_reach_decay": self.near_reach_decay,
            "grasp_acquire_close": self.grasp_acquire_close,
            "grasp_acquire_gate": self.grasp_acquire_gate,
        }
        if any(
            not 0.0 <= float(value) <= 1.0
            for value in unit_values.values()
        ):
            raise ValueError(
                "Staged reward unit values must lie in [0, 1]: "
                f"{unit_values}."
            )
        if not 0.0 <= float(self.contact_gate_threshold) <= 1.0:
            raise ValueError("contact_gate_threshold must lie in [0, 1].")
        if not 0.0 <= float(self.center_proximity_close_threshold) <= 1.0:
            raise ValueError(
                "center_proximity_close_threshold must lie in [0, 1]."
            )
        if float(self.center_proximity_limit_m) < 0.0:
            raise ValueError("center_proximity_limit_m must be non-negative.")
        if (
            float(self.center_proximity_penalty) > 0.0
            and float(self.center_proximity_limit_m) <= 0.0
        ):
            raise ValueError(
                "center_proximity_penalty requires a positive "
                "center_proximity_limit_m."
            )
        object.__setattr__(
            self,
            "require_bilateral_contact_for_lift_reward",
            bool(self.require_bilateral_contact_for_lift_reward),
        )
        object.__setattr__(
            self,
            "ungated_negative_lift_progress",
            bool(self.ungated_negative_lift_progress),
        )
        for name in (
            "gripper_reward_requires_contact",
            "grasp_acquire_requires_contact",
            "require_generic_contact_for_lift",
            "contact_reward_onset_only",
            "contact_missing_requires_previous",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RewardShapingConfig":
        return cls(**dict(values))


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 200_000
    # Four workers keep the default usable on a shared workstation. MuJoCo
    # models are process-local, so memory scales with this value; larger runs
    # must opt in after checking live available memory.
    num_envs: int = 4
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    # Safety projection makes effective-action rollouts mildly off-policy;
    # bound the update drift and importance ratios before they can overflow.
    target_kl: float = 0.03
    max_log_ratio: float = 10.0
    advantage_clip: float = 10.0
    value_coef: float = 0.5
    # Optional critic trust region around the rollout-time prediction.  The
    # established shared-backbone baseline remains the default because 0.2
    # clipping did not improve matched CPU Reach probes consistently.
    value_clip_coef: float = 0.0
    entropy_coef: float = 0.0
    # Optional trust region around the actor captured after checkpoint/
    # bootstrap initialization. It uses only normalized deployable actor
    # observations and is disabled by default for legacy PPO behavior.
    actor_reference_mse_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    seed: int = 0
    checkpoint_interval: int = 50_000
    start_method: str = "spawn"
    device: str = "auto"
    initial_log_std: float = -0.5
    final_log_std: float = -2.5
    std_anneal_fraction: float = 0.75
    # When a checkpoint grows a named action head, do not inject the fresh
    # run's broad exploration into that new physical component. Shared
    # components preserve checkpoint std; a new component starts no broader
    # than this cap and then follows the ordinary anneal. Fresh runs are
    # unchanged.
    warm_start_new_action_log_std_cap: float = -1.5
    # ``actor`` is the deployable default. ``privileged_training_state`` is an
    # opt-in asymmetric critic that may consume simulator-only 31-D state;
    # actor inputs and actor inference remain unchanged.
    critic_observation_mode: str = "actor"
    # Same-observation training shares the actor representation by default:
    # matched Reach runs learned materially faster than a fully independent
    # critic.  ``separate`` remains an explicit isolation ablation and is
    # always selected for the differently shaped privileged critic.
    critic_backbone_mode: str = "shared_actor"
    # Retained only so old CLI/config payloads fail with a precise migration
    # message. Replacing a sampled latent action with a projected or measured
    # downstream action violates PPO's on-policy log-probability contract.
    effective_action_for_ppo: bool = False

    def __post_init__(self) -> None:
        positive_ints = {
            "total_timesteps": self.total_timesteps,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "hidden_dim": self.hidden_dim,
            "checkpoint_interval": self.checkpoint_interval,
        }
        invalid = {name: value for name, value in positive_ints.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"PPO positive integer settings required: {invalid}.")
        if self.minibatch_size > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed one rollout batch.")
        unit_interval = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
        }
        if any(not 0.0 < float(value) <= 1.0 for value in unit_interval.values()):
            raise ValueError(f"PPO unit-interval settings invalid: {unit_interval}.")
        positive_floats = {
            "target_kl": self.target_kl,
            "max_log_ratio": self.max_log_ratio,
            "advantage_clip": self.advantage_clip,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive_floats.values()
        ):
            raise ValueError(
                "PPO numerical-stability settings must be finite and positive: "
                f"{positive_floats}."
            )
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive.")
        if (
            not math.isfinite(float(self.value_clip_coef))
            or float(self.value_clip_coef) < 0.0
        ):
            raise ValueError("value_clip_coef must be finite and non-negative.")
        if (
            not math.isfinite(float(self.actor_reference_mse_coef))
            or float(self.actor_reference_mse_coef) < 0.0
        ):
            raise ValueError(
                "actor_reference_mse_coef must be finite and non-negative."
            )
        if not math.isfinite(float(self.initial_log_std)) or not math.isfinite(
            float(self.final_log_std)
        ):
            raise ValueError("PPO log standard deviations must be finite.")
        if self.final_log_std > self.initial_log_std:
            raise ValueError(
                "final_log_std cannot exceed initial_log_std for exploration annealing."
            )
        if not 0.0 < float(self.std_anneal_fraction) <= 1.0:
            raise ValueError("std_anneal_fraction must lie in (0, 1].")
        warm_start_cap = float(self.warm_start_new_action_log_std_cap)
        if not math.isfinite(warm_start_cap) or not -5.0 <= warm_start_cap <= 1.0:
            raise ValueError(
                "warm_start_new_action_log_std_cap must be finite and lie "
                "within the policy clamp [-5, 1]."
            )
        object.__setattr__(
            self,
            "warm_start_new_action_log_std_cap",
            warm_start_cap,
        )
        critic_observation_mode = str(self.critic_observation_mode).strip().lower()
        if critic_observation_mode not in {
            "actor",
            "privileged_training_state",
        }:
            raise ValueError(
                "critic_observation_mode must be 'actor' or "
                "'privileged_training_state'."
            )
        object.__setattr__(
            self,
            "critic_observation_mode",
            critic_observation_mode,
        )
        critic_backbone_mode = str(self.critic_backbone_mode).strip().lower()
        if critic_backbone_mode not in {"separate", "shared_actor"}:
            raise ValueError(
                "critic_backbone_mode must be 'separate' or 'shared_actor'."
            )
        if critic_observation_mode == "privileged_training_state":
            # The privileged input has a different contract and cannot share
            # the actor encoder.  Normalize the recorded topology so opting in
            # to privileged training never changes the deployment contract or
            # requires a second architecture flag.
            critic_backbone_mode = "separate"
        object.__setattr__(
            self,
            "critic_backbone_mode",
            critic_backbone_mode,
        )
        effective_action_for_ppo = bool(self.effective_action_for_ppo)
        if effective_action_for_ppo:
            raise ValueError(
                "effective_action_for_ppo is no longer supported: PPO must "
                "store its sampled latent action; task conditioning and "
                "safety admission are downstream transition dynamics."
            )
        object.__setattr__(self, "effective_action_for_ppo", False)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PPOConfig":
        payload = dict(values)
        # Missing fields identify a pre-isolation checkpoint/config. Preserve
        # its shared/unclipped optimization semantics explicitly so future
        # default experiments cannot reinterpret an old checkpoint.
        payload.setdefault("value_clip_coef", 0.0)
        payload.setdefault("critic_backbone_mode", "shared_actor")
        return cls(**payload)


@dataclass(frozen=True)
class GoalDeltaActorBootstrapConfig:
    """Optional deployable-observation actor initialization before PPO.

    A zero transition count is the default and leaves PPO initialization
    unchanged.  When enabled, actor-visible reach-target delta (and, for the
    Contact/Pick bridge, measured gripper close) is converted into the task's
    latent action. The optional Pick teacher may use simulator contact phase to
    label lift transitions, but the fitted actor still receives only deployable
    observations and the critic is never initialized from teacher state.
    """

    transitions: int = 0
    epochs: int = 80
    minibatch_size: int = 256
    learning_rate: float = 1e-3
    target_mse: float = 1e-4
    max_grad_norm: float = 1.0
    lift_step_m: float = 0.004
    pick_close_distance_m: float = 0.035
    pick_lift_lock_xy: bool = False

    def __post_init__(self) -> None:
        if int(self.transitions) < 0:
            raise ValueError("bootstrap transitions must be non-negative.")
        positive_ints = {
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
        }
        if any(int(value) <= 0 for value in positive_ints.values()):
            raise ValueError(
                "bootstrap epochs and minibatch_size must be positive."
            )
        positive_floats = {
            "learning_rate": self.learning_rate,
            "target_mse": self.target_mse,
            "max_grad_norm": self.max_grad_norm,
            "lift_step_m": self.lift_step_m,
            "pick_close_distance_m": self.pick_close_distance_m,
        }
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in positive_floats.values()
        ):
            raise ValueError(
                "bootstrap learning_rate, target_mse, and max_grad_norm "
                "must be finite and positive."
            )

    @property
    def enabled(self) -> bool:
        return int(self.transitions) > 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "enabled": self.enabled}


__all__ = [
    "GoalDeltaActorBootstrapConfig",
    "PPOConfig",
    "RewardShapingConfig",
]
