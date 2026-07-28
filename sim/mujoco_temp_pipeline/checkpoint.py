"""Portable checkpoint boundary for the temporary policy."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

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


def check_task_config_compatibility(
    payload: Mapping[str, Any],
    task_config_path: str | Path,
    *,
    allow_mismatch: bool = False,
    allow_cube_pose_observation_expansion: bool = False,
) -> dict[str, object]:
    """Validate the task boundary, with an explicit diagnostics-only escape hatch."""

    expected_hash = payload.get("task_config_sha256")
    actual_hash = file_sha256(task_config_path)
    mismatch = expected_hash != actual_hash
    if mismatch and not allow_mismatch:
        raise ValueError(
            "Task config does not match checkpoint: "
            f"expected {expected_hash}, got {actual_hash}. "
            "Use --allow-task-config-mismatch only for legacy collision diagnostics."
        )
    checkpoint_metadata = payload.get("task_metadata")
    observation_expansion_migration = False
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
                    raise ValueError(
                        "Learning contract does not match checkpoint: "
                        f"field={key}, checkpoint={expected!r}, runtime={actual!r}."
                    )
    return {
        "task_config_mismatch": mismatch,
        "checkpoint_task_config_sha256": expected_hash,
        "runtime_task_config_sha256": actual_hash,
        "cube_pose_observation_expansion": observation_expansion_migration,
    }


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


def save_checkpoint(
    path: str | Path,
    *,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    observation_normalizer: RunningMeanStd,
    ppo_config: PPOConfig,
    reward_config: RewardShapingConfig,
    global_step: int,
    update: int,
    task_config_path: str | Path,
) -> Path:
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
            "state_dict": policy.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "observation_normalizer": observation_normalizer.to_state(),
        "ppo_config": ppo_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "task_config_path": str(task_path),
        "task_config_sha256": file_sha256(task_path),
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
    "check_task_config_compatibility",
    "expand_cube_pose_policy_state",
    "scale_pose_action_head_for_doubled_step",
    "file_sha256",
    "load_checkpoint_payload",
    "save_checkpoint",
]
