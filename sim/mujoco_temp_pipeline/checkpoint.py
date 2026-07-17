"""Portable checkpoint boundary for the temporary policy."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import torch

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
    return {
        "task_config_mismatch": mismatch,
        "checkpoint_task_config_sha256": expected_hash,
        "runtime_task_config_sha256": actual_hash,
    }


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
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "global_step": int(global_step),
        "update": int(update),
        "policy": {
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "hidden_dim": policy.hidden_dim,
            "state_dict": policy.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "observation_normalizer": observation_normalizer.to_state(),
        "ppo_config": ppo_config.to_dict(),
        "reward_config": reward_config.to_dict(),
        "task_config_path": str(task_path),
        "task_config_sha256": file_sha256(task_path),
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
    "file_sha256",
    "load_checkpoint_payload",
    "save_checkpoint",
]
