"""Portable checkpoint boundary for the temporary policy."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

from .config import PPOConfig, RewardShapingConfig
from .normalization import RunningMeanStd
from .policy import ActorCritic


CHECKPOINT_SCHEMA_VERSION = "dexteleop.sim.mujoco_temp_ppo.v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_config_sha256(path: str | Path) -> str:
    """Hash a task profile and any inherited base profile as one contract."""

    return CubeLiftTaskConfig.load(path).source_sha256


def check_task_config_compatibility(
    payload: Mapping[str, Any],
    task_config_path: str | Path,
    *,
    allow_mismatch: bool = False,
    allow_cube_pose_observation_expansion: bool = False,
    allow_policy_action_component_transfer: bool = False,
    allow_policy_action_transform_transfer: bool = False,
    allow_task_action_conditioning_transfer: bool = False,
) -> dict[str, object]:
    """Validate exact reuse or one explicitly enabled migration boundary."""

    expected_hash = payload.get("task_config_sha256")
    actual_hash = task_config_sha256(task_config_path)
    mismatch = expected_hash != actual_hash
    if mismatch and not allow_mismatch:
        raise ValueError(
            "Task config does not match checkpoint: "
            f"expected {expected_hash}, got {actual_hash}. "
            "Use --allow-task-config-mismatch only for legacy collision diagnostics."
        )
    checkpoint_metadata = payload.get("task_metadata")
    observation_expansion_migration = False
    legacy_compact_action_contract_migration = False
    policy_action_component_transfer: dict[str, object] | None = None
    policy_action_transform_transfer: dict[str, object] | None = None
    task_action_conditioning_transfer: dict[str, object] | None = None
    if isinstance(checkpoint_metadata, Mapping):
        runtime_task_config = CubeLiftTaskConfig.load(task_config_path)
        checkpoint_learning = checkpoint_metadata.get("learning_contract")
        runtime_learning = build_learning_metadata(runtime_task_config)
        if isinstance(checkpoint_learning, Mapping):
            observation_expansion_migration = (
                allow_cube_pose_observation_expansion
                and _is_cube_pose_observation_expansion(
                    checkpoint_learning,
                    runtime_learning,
                )
            )
            legacy_compact_action_contract_migration = (
                _is_legacy_compact_action_contract(
                    payload,
                    checkpoint_learning,
                    runtime_learning,
                    runtime_task_config,
                )
            )
            if allow_policy_action_component_transfer:
                policy_action_component_transfer = (
                    _policy_action_component_transfer(
                        checkpoint_learning,
                        runtime_learning,
                    )
                )
            if allow_policy_action_transform_transfer:
                policy_action_transform_transfer = (
                    _policy_action_transform_transfer(
                        checkpoint_learning,
                        runtime_learning,
                    )
                )
            if allow_task_action_conditioning_transfer:
                task_action_conditioning_transfer = (
                    _task_action_conditioning_transfer(
                        checkpoint_learning,
                        runtime_learning,
                    )
                )
        checkpoint_action_layout = checkpoint_metadata.get("action_layout")
        if checkpoint_action_layout is not None:
            runtime_action_layout = runtime_task_config.action_layout
            if str(checkpoint_action_layout) != str(runtime_action_layout):
                raise ValueError(
                    "Task action layout does not match checkpoint: "
                    f"checkpoint={checkpoint_action_layout!r}, "
                    f"runtime={runtime_action_layout!r}. Retrain the policy; "
                    "a qpos checkpoint cannot be reused as a pose policy."
                )
        checkpoint_observation_layout = checkpoint_metadata.get("observation_layout")
        if checkpoint_observation_layout is not None:
            runtime_observation_layout = runtime_task_config.observation_layout
            if (
                str(checkpoint_observation_layout) != str(runtime_observation_layout)
                and not observation_expansion_migration
            ):
                raise ValueError(
                    "Task observation layout does not match checkpoint: "
                    f"checkpoint={checkpoint_observation_layout!r}, "
                    f"runtime={runtime_observation_layout!r}. Retrain the policy; "
                    "a state-only checkpoint cannot be reused as a wrist-RGB policy."
                )
        if isinstance(checkpoint_learning, Mapping):
            for key in (
                "model_manifest_version",
                "policy_class",
                "observation_contract_hash",
                "model_input_binding_hash",
                "action_binding_hash",
                "model_input_dim",
                "model_output_dim",
                "policy_action_transform_hash",
                "task_action_binding_hash",
                "task_action_conditioning_hash",
            ):
                expected = checkpoint_learning.get(key)
                actual = runtime_learning.get(key)
                if expected is not None and str(expected) != str(actual):
                    if (
                        observation_expansion_migration
                        and key
                        in {
                            "observation_contract_hash",
                            "model_input_binding_hash",
                            "model_input_dim",
                        }
                    ):
                        continue
                    if (
                        legacy_compact_action_contract_migration
                        and key in {"action_binding_hash", "model_output_dim"}
                    ):
                        continue
                    if (
                        policy_action_component_transfer is not None
                        and key
                        in {
                            "action_binding_hash",
                            "model_output_dim",
                            "policy_action_transform_hash",
                        }
                    ):
                        continue
                    if (
                        policy_action_transform_transfer is not None
                        and key == "policy_action_transform_hash"
                    ):
                        continue
                    if (
                        task_action_conditioning_transfer is not None
                        and key == "task_action_conditioning_hash"
                    ):
                        continue
                    raise ValueError(
                        "Learning contract does not match checkpoint: "
                        f"field={key}, checkpoint={expected!r}, runtime={actual!r}."
                    )
    return {
        "task_config_mismatch": mismatch,
        "checkpoint_task_config_sha256": expected_hash,
        "runtime_task_config_sha256": actual_hash,
        "cube_pose_observation_expansion": observation_expansion_migration,
        "legacy_compact_action_contract_migration": (
            legacy_compact_action_contract_migration
        ),
        "policy_action_component_transfer": policy_action_component_transfer,
        "policy_action_transform_transfer": policy_action_transform_transfer,
        "task_action_conditioning_transfer": (
            task_action_conditioning_transfer
        ),
    }


def _policy_action_component_transfer(
    checkpoint_learning: Mapping[str, Any],
    runtime_learning: Mapping[str, Any],
) -> dict[str, object] | None:
    """Describe a safe named-output-head transfer across task stages."""

    checkpoint_components = tuple(
        str(value)
        for value in checkpoint_learning.get("policy_action_components", ())
    )
    runtime_components = tuple(
        str(value)
        for value in runtime_learning.get("policy_action_components", ())
    )
    if not checkpoint_components or not runtime_components:
        return None
    if checkpoint_components == runtime_components:
        return None
    if (
        checkpoint_learning.get("task_action_binding_hash")
        != runtime_learning.get("task_action_binding_hash")
    ):
        return None
    shared = tuple(
        component
        for component in runtime_components
        if component in checkpoint_components
    )
    if not shared:
        return None
    return {
        "checkpoint_components": checkpoint_components,
        "runtime_components": runtime_components,
        "shared_components": shared,
        "new_components": tuple(
            component
            for component in runtime_components
            if component not in checkpoint_components
        ),
        "dropped_components": tuple(
            component
            for component in checkpoint_components
            if component not in runtime_components
        ),
    }


def _policy_action_transform_transfer(
    checkpoint_learning: Mapping[str, Any],
    runtime_learning: Mapping[str, Any],
) -> dict[str, object] | None:
    """Describe an actor-only warm start across a changed task decoder.

    This migration keeps the latent labels, actor input, and public task-action
    binding identical.  It is intentionally narrower than task mismatch and
    is only enabled by the training warm-start path; evaluation and deployment
    continue to require an exact transform hash.
    """

    invariant_keys = (
        "model_manifest_version",
        "policy_class",
        "observation_contract_hash",
        "model_input_binding_hash",
        "action_binding_hash",
        "model_input_dim",
        "model_output_dim",
        "task_action_binding_hash",
    )
    if any(
        checkpoint_learning.get(key) != runtime_learning.get(key)
        for key in invariant_keys
    ):
        return None
    checkpoint_components = tuple(
        str(value)
        for value in checkpoint_learning.get("policy_action_components", ())
    )
    runtime_components = tuple(
        str(value)
        for value in runtime_learning.get("policy_action_components", ())
    )
    checkpoint_hash = checkpoint_learning.get("policy_action_transform_hash")
    runtime_hash = runtime_learning.get("policy_action_transform_hash")
    if (
        not checkpoint_components
        or checkpoint_components != runtime_components
        or checkpoint_hash is None
        or runtime_hash is None
        or checkpoint_hash == runtime_hash
    ):
        return None
    return {
        "policy_action_components": checkpoint_components,
        "checkpoint_transform_id": checkpoint_learning.get(
            "policy_action_transform_id"
        ),
        "checkpoint_transform_hash": checkpoint_hash,
        "runtime_transform_id": runtime_learning.get(
            "policy_action_transform_id"
        ),
        "runtime_transform_hash": runtime_hash,
        "critic_reset_required": True,
    }


def _task_action_conditioning_transfer(
    checkpoint_learning: Mapping[str, Any],
    runtime_learning: Mapping[str, Any],
) -> dict[str, object] | None:
    """Describe actor-only reuse across changed environment action dynamics.

    The actor observation, latent labels, and public task-action binding must
    remain identical. The conditioner itself is deliberately not declared
    compatible: this receipt authorizes only a training warm start and requires
    a fresh critic for the changed transition dynamics.
    """

    invariant_keys = (
        "model_manifest_version",
        "policy_class",
        "observation_contract_hash",
        "model_input_binding_hash",
        "action_binding_hash",
        "model_input_dim",
        "model_output_dim",
        "task_action_binding_hash",
        "policy_action_components",
    )
    if any(
        checkpoint_learning.get(key) != runtime_learning.get(key)
        for key in invariant_keys
    ):
        return None
    checkpoint_hash = checkpoint_learning.get(
        "task_action_conditioning_hash"
    )
    runtime_hash = runtime_learning.get("task_action_conditioning_hash")
    if (
        checkpoint_hash is None
        or runtime_hash is None
        or checkpoint_hash == runtime_hash
    ):
        return None
    return {
        "checkpoint_conditioning_hash": checkpoint_hash,
        "runtime_conditioning_hash": runtime_hash,
        "critic_reset_required": True,
    }


def _is_legacy_compact_action_contract(
    payload: Mapping[str, Any],
    checkpoint_learning: Mapping[str, Any],
    runtime_learning: Mapping[str, Any],
    runtime_task_config: CubeLiftTaskConfig,
) -> bool:
    """Recognize checkpoints whose old metadata hid a compact actor behind 7-D."""

    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        return False
    if checkpoint_learning.get("policy_action_transform_hash") is not None:
        return False
    expected_indices = tuple(runtime_task_config.policy_active_action_indices)
    checkpoint_indices = tuple(
        int(value)
        for value in policy.get("active_action_indices", ())
    )
    return bool(
        runtime_task_config.policy_action_dim < runtime_task_config.action_dim
        and int(policy.get("action_dim", -1))
        == runtime_task_config.policy_action_dim
        and int(policy.get("environment_action_dim", -1))
        == runtime_task_config.action_dim
        and checkpoint_indices == expected_indices
        and str(checkpoint_learning.get("model_output_dim"))
        == str(runtime_task_config.action_dim)
        and checkpoint_learning.get("action_binding_hash")
        == runtime_learning.get("task_action_binding_hash")
    )


def _is_cube_pose_observation_expansion(
    checkpoint_learning: Mapping[str, Any],
    runtime_learning: Mapping[str, Any],
) -> bool:
    """Recognize only the additive v4 (28-D) -> v5 (35-D) state migration."""

    return bool(
        checkpoint_learning.get("policy_class") == "normal"
        and runtime_learning.get("policy_class") == "normal"
        and checkpoint_learning.get("model_input_dim") == 28
        and runtime_learning.get("model_input_dim") == 35
        and str(checkpoint_learning.get("model_output_dim")) == "7"
        and str(runtime_learning.get("model_output_dim")) == "7"
        and str(checkpoint_learning.get("action_binding_id", "")).endswith(
            "right_tcp_delta_xyz3_rotvec3_right_gripper_delta1.v1"
        )
        and checkpoint_learning.get("action_binding_hash")
        == runtime_learning.get("action_binding_hash")
        and str(checkpoint_learning.get("observation_contract_id", "")).endswith(
            "right_target_error6_right_gripper_target_error1_last_action7.v2"
        )
        and str(runtime_learning.get("observation_contract_id", "")).endswith(
            "right_target_error6_right_gripper_target_error1_last_action7_cube_pose7.v1"
        )
    )


def expand_cube_pose_policy_state(
    source_policy: ActorCritic,
    target_policy: ActorCritic,
) -> None:
    """Copy a 28-D normal actor into the additive 35-D cube-pose actor.

    Only the first input layer changes shape.  The seven new cube-pose columns
    start at zero so the warm-start policy initially preserves its old motion;
    PPO can then learn a non-zero use of the newly exposed pose features.
    """

    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    source_weight = source_state.get("backbone.0.weight")
    target_weight = target_state.get("backbone.0.weight")
    if source_weight is None or target_weight is None:
        raise ValueError("Cube-pose migration requires a plain proprio backbone.")
    if (
        source_weight.ndim != 2
        or target_weight.ndim != 2
        or target_weight.shape[0] != source_weight.shape[0]
        or target_weight.shape[1] != source_weight.shape[1] + 7
    ):
        raise ValueError(
            "Unsupported cube-pose policy expansion: "
            f"source={tuple(source_weight.shape)}, target={tuple(target_weight.shape)}."
        )
    for name, target_value in target_state.items():
        if name.startswith("critic"):
            # Observation/task migrations preserve actor behavior only.  The
            # value function is task-specific and may also use a different,
            # optional privileged input topology.
            continue
        source_value = source_state.get(name)
        if source_value is None:
            raise ValueError(f"Checkpoint is missing policy parameter {name!r}.")
        if name == "backbone.0.weight":
            expanded = target_value.detach().clone()
            expanded.zero_()
            expanded[:, : source_value.shape[1]] = source_value
            target_state[name] = expanded
        elif target_value.shape != source_value.shape:
            raise ValueError(
                f"Unsupported cube-pose parameter shape change for {name!r}: "
                f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}."
            )
        else:
            target_state[name] = source_value
    target_policy.load_state_dict(target_state)
    target_policy.reset_critic()


def scale_pose_action_head_for_doubled_step(
    policy: ActorCritic,
    *,
    pre_tanh_scale: float = 0.25,
) -> None:
    """Make a v4 actor a conservative transfer policy for v5's larger step.

    The v5 task doubles the physical TCP/rotation increment while keeping the
    normalized action contract unchanged.  A literal v4 actor would therefore
    produce roughly twice the physical motion on its first rollout, often
    entering the collision projector before it has a chance to adapt.  Scale
    only the six arm outputs (the gripper output remains unchanged).  PPO can
    subsequently grow the actor head again as the new 35-D observation is
    learned; the normal PPO exploration schedule remains authoritative.

    ``pre_tanh_scale`` is applied before the actor's tanh, so 0.25 is a
    deliberately conservative transfer value rather than a claim that tanh
    outputs are exactly linear.
    """

    scale = float(pre_tanh_scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("pre_tanh_scale must be finite and lie in (0, 1].")
    if policy.action_dim < 6:
        raise ValueError("A pose action policy must expose at least six arm outputs.")
    with torch.no_grad():
        policy.actor_mean.weight[:6].mul_(scale)
        policy.actor_mean.bias[:6].mul_(scale)


def transfer_policy_action_components(
    source_policy: ActorCritic,
    target_policy: ActorCritic,
    *,
    source_components: tuple[str, ...],
    target_components: tuple[str, ...],
) -> tuple[str, ...]:
    """Transfer shared actor outputs by semantic component name.

    The target keeps freshly initialized rows for new components. The critic
    head is reset because task objective/reward semantics may have changed.
    Observation encoders and the shared backbone must match exactly.
    """

    source_components = tuple(str(value) for value in source_components)
    target_components = tuple(str(value) for value in target_components)
    if len(source_components) != source_policy.action_dim:
        raise ValueError("Source action components do not match source actor.")
    if len(target_components) != target_policy.action_dim:
        raise ValueError("Target action components do not match target actor.")
    if len(source_components) != len(set(source_components)) or len(
        target_components
    ) != len(set(target_components)):
        raise ValueError("Action component names must be unique.")
    shared = tuple(
        component
        for component in target_components
        if component in source_components
    )
    if not shared:
        raise ValueError("Action component transfer requires a shared component.")

    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    output_keys = {"actor_mean.weight", "actor_mean.bias", "log_std"}
    for name, target_value in target_state.items():
        if name in output_keys or name.startswith("critic"):
            continue
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            raise ValueError(
                "Policy encoder/backbone mismatch during action component "
                f"transfer: field={name!r}, source="
                f"{None if source_value is None else tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}."
            )
        target_state[name] = source_value

    for component in shared:
        source_index = source_components.index(component)
        target_index = target_components.index(component)
        target_state["actor_mean.weight"][target_index] = source_state[
            "actor_mean.weight"
        ][source_index]
        target_state["actor_mean.bias"][target_index] = source_state[
            "actor_mean.bias"
        ][source_index]
        target_state["log_std"][target_index] = source_state["log_std"][
            source_index
        ]
    target_policy.load_state_dict(target_state)
    target_policy.reset_critic()
    return shared


def build_warm_start_exploration_schedule(
    *,
    source_log_std: Any,
    source_components: tuple[str, ...],
    target_components: tuple[str, ...],
    configured_initial_log_std: float,
    configured_final_log_std: float,
    new_action_log_std_cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve mature component noise and bound newly added exploration.

    The result is ordered like ``target_components``. Shared named components
    start from their checkpoint log std. New components start from the tighter
    of the fresh-run initial value and ``new_action_log_std_cap``. No component
    is ever scheduled toward a broader distribution than its warm-start value.
    """

    source_names = tuple(str(value) for value in source_components)
    target_names = tuple(str(value) for value in target_components)
    source_values = np.asarray(source_log_std, dtype=np.float32).reshape(-1)
    if source_values.shape != (len(source_names),):
        raise ValueError(
            "Source log std must match source action components: "
            f"values={source_values.shape}, components={len(source_names)}."
        )
    if (
        not source_names
        or not target_names
        or len(source_names) != len(set(source_names))
        or len(target_names) != len(set(target_names))
    ):
        raise ValueError("Warm-start action component names must be unique and non-empty.")
    scalar_values = np.asarray(
        (
            configured_initial_log_std,
            configured_final_log_std,
            new_action_log_std_cap,
        ),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(source_values)) or not np.all(np.isfinite(scalar_values)):
        raise ValueError("Warm-start exploration values must be finite.")
    if np.any(source_values < -5.0) or np.any(source_values > 1.0):
        raise ValueError("Checkpoint log std must lie within [-5, 1].")
    if np.any(scalar_values < -5.0) or np.any(scalar_values > 1.0):
        raise ValueError("Configured warm-start log std must lie within [-5, 1].")
    if configured_final_log_std > configured_initial_log_std:
        raise ValueError("Configured final log std cannot exceed initial log std.")

    source_by_component = dict(zip(source_names, source_values, strict=True))
    new_component_initial = min(
        float(configured_initial_log_std),
        float(new_action_log_std_cap),
    )
    initial = np.asarray(
        [
            source_by_component.get(component, new_component_initial)
            for component in target_names
        ],
        dtype=np.float32,
    )
    final = np.minimum(
        np.full(len(target_names), float(configured_final_log_std), dtype=np.float32),
        initial,
    )
    return initial, final


def transfer_actor_state(
    source_policy: ActorCritic,
    target_policy: ActorCritic,
) -> None:
    """Copy an exactly compatible actor while resetting any critic topology.

    This is the explicit migration used when toggling the optional asymmetric
    critic.  It never copies ``critic`` or ``critic_backbone`` parameters, so
    a simulator-only value input cannot alter the deployable actor state.
    """

    if (
        source_policy.observation_dim != target_policy.observation_dim
        or source_policy.action_dim != target_policy.action_dim
        or source_policy.environment_action_dim
        != target_policy.environment_action_dim
        or source_policy.active_action_indices
        != target_policy.active_action_indices
    ):
        raise ValueError("Actor-state transfer requires identical model I/O mappings.")
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    for name, target_value in target_state.items():
        if name.startswith("critic"):
            continue
        source_value = source_state.get(name)
        if source_value is None or source_value.shape != target_value.shape:
            raise ValueError(
                "Actor-state transfer parameter mismatch: "
                f"field={name!r}, source="
                f"{None if source_value is None else tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}."
            )
        target_state[name] = source_value
    target_policy.load_state_dict(target_state)
    target_policy.reset_critic()


def save_checkpoint(
    path: str | Path,
    *,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observation_normalizer: RunningMeanStd,
    critic_observation_normalizer: RunningMeanStd | None = None,
    ppo_config: PPOConfig,
    reward_config: RewardShapingConfig,
    global_step: int,
    update: int,
    task_config_path: str | Path,
    policy_initialization: Mapping[str, Any] | None = None,
) -> Path:
    expected_separate_critic = (
        ppo_config.critic_backbone_mode == "separate"
        or ppo_config.critic_observation_mode
        == "privileged_training_state"
    )
    if policy.separate_critic_backbone != expected_separate_critic:
        raise ValueError(
            "Policy critic topology does not match PPOConfig: "
            f"policy_separate={policy.separate_critic_backbone}, "
            f"config_mode={ppo_config.critic_backbone_mode!r}."
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    task_path = Path(task_config_path).resolve()
    task_config = CubeLiftTaskConfig.load(task_path)
    task_metadata = task_config.to_metadata()
    task_metadata["learning_contract"] = build_learning_metadata(task_config)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "global_step": int(global_step),
        "update": int(update),
        "policy": {
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "environment_action_dim": policy.environment_action_dim,
            "active_action_indices": list(policy.active_action_indices),
            "trainable_log_std": policy.trainable_log_std,
            "hidden_dim": policy.hidden_dim,
            "proprio_dim": policy.proprio_dim,
            "image_shape": (
                None if policy.image_shape is None else list(policy.image_shape)
            ),
            "image_pool_size": int(policy.image_pool_size),
            "critic_observation_dim": int(policy.critic_observation_dim),
            "separate_critic_backbone": bool(
                policy.separate_critic_backbone
            ),
            "critic_observation_mode": ppo_config.critic_observation_mode,
            "state_dict": policy.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "observation_normalizer": observation_normalizer.to_state(),
        "critic_observation_normalizer": (
            None
            if critic_observation_normalizer is None
            else critic_observation_normalizer.to_state()
        ),
        "ppo_config": ppo_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "policy_initialization": (
            None
            if policy_initialization is None
            else dict(policy_initialization)
        ),
        "task_config_path": str(task_path),
        "task_config_sha256": task_config.source_sha256,
        "task_metadata": task_metadata,
        # Cast TorchVersion to plain str so weights_only=True can load the
        # checkpoint without allowlisting a Python class.
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported temporary PPO checkpoint: {path}.")
    return payload


def build_policy_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[ActorCritic, RunningMeanStd, dict[str, Any]]:
    resolved_device = torch.device(device)
    payload = load_checkpoint_payload(path, map_location=resolved_device)
    policy_data = _require_mapping(payload, "policy")
    policy = ActorCritic(
        observation_dim=int(policy_data["observation_dim"]),
        action_dim=int(policy_data["action_dim"]),
        hidden_dim=int(policy_data["hidden_dim"]),
        proprio_dim=(
            None
            if policy_data.get("proprio_dim") is None
            else int(policy_data["proprio_dim"])
        ),
        image_shape=(
            None
            if policy_data.get("image_shape") is None
            else tuple(int(value) for value in policy_data["image_shape"])
        ),
        image_pool_size=int(policy_data.get("image_pool_size", 1)),
        critic_observation_dim=int(
            policy_data.get(
                "critic_observation_dim",
                policy_data["observation_dim"],
            )
        ),
        separate_critic_backbone=bool(
            policy_data.get("separate_critic_backbone", False)
        ),
        environment_action_dim=int(
            policy_data.get(
                "environment_action_dim",
                policy_data["action_dim"],
            )
        ),
        active_action_indices=tuple(
            int(value)
            for value in policy_data.get(
                "active_action_indices",
                range(int(policy_data["action_dim"])),
            )
        ),
        trainable_log_std=bool(
            policy_data.get("trainable_log_std", True)
        ),
    ).to(resolved_device)
    policy.load_state_dict(_require_mapping(policy_data, "state_dict"))
    policy.eval()
    normalizer = RunningMeanStd.from_state(
        _require_mapping(payload, "observation_normalizer")
    )
    return policy, normalizer, payload


def _require_mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = values.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Checkpoint field {key!r} must be a mapping.")
    return result


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "build_policy_from_checkpoint",
    "build_warm_start_exploration_schedule",
    "check_task_config_compatibility",
    "expand_cube_pose_policy_state",
    "scale_pose_action_head_for_doubled_step",
    "file_sha256",
    "load_checkpoint_payload",
    "save_checkpoint",
    "task_config_sha256",
    "transfer_actor_state",
    "transfer_policy_action_components",
]
