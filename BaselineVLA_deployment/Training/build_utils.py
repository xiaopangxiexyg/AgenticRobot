from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.med_support import normalize_med_spec


def _resolve_reference_split_summary(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    for split_name in ("train", "val"):
        split_summary = data_summary.get(split_name)
        if split_summary is not None:
            return split_summary
    raise KeyError("data_summary must include at least one split summary under `train` or `val`.")


def resolve_model_family(config: Any) -> str:
    return str(getattr(config.model, "family", "mlp")).lower()


def resolve_action_chunk(data_summary: Dict[str, Any]) -> int:
    split_summary = _resolve_reference_split_summary(data_summary)
    return int(split_summary.get("action_chunk", split_summary.get("action_horizon", 1)))


def resolve_state_dim(data_summary: Dict[str, Any]) -> int:
    split_summary = _resolve_reference_split_summary(data_summary)
    return int(split_summary["state_dim"])


def resolve_action_dim(data_summary: Dict[str, Any]) -> int:
    split_summary = _resolve_reference_split_summary(data_summary)
    return int(split_summary["action_dim"])


def resolve_train_dimensions(data_summary: Dict[str, Any]) -> tuple[int, int, int]:
    state_dim = resolve_state_dim(data_summary)
    action_dim = resolve_action_dim(data_summary)
    action_chunk = resolve_action_chunk(data_summary)
    return state_dim, action_dim, action_chunk


def resolve_tactile_observation_dim(data_summary: Dict[str, Any]) -> int:
    split_summary = _resolve_reference_split_summary(data_summary)
    tactile_dim = split_summary.get("tactile_observation_dim", None)
    if tactile_dim is None:
        shape = split_summary.get("tactile_observation_shape", None)
        if shape:
            tactile_dim = list(shape)[-1]
    if tactile_dim is None:
        tactile_dim = split_summary.get("tactile_dim", None)
    if tactile_dim is None:
        raise ValueError(
            "Unable to infer tactile observation dim. "
            "Enable raw.sequence.tactile in the data config or provide tactile_observation_dim in data summary."
        )
    return int(tactile_dim)


def resolve_med_dim(config: Any, data_summary: Dict[str, Any], experiment_name: str) -> int:
    return int(resolve_med_spec(config, data_summary, experiment_name)["flat_dim"])


def resolve_med_spec(config: Any, data_summary: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    data_config = getattr(config, "data", None)
    data_med_config = None if data_config is None else getattr(data_config, "med", None)
    split_summary = _resolve_reference_split_summary(data_summary)
    med_spec_value = data_summary.get("med_spec") or split_summary.get("med_spec")
    if med_spec_value is not None:
        return normalize_med_spec(med_spec_value)

    med_dim_value = data_summary.get("med_dim") or split_summary.get("med_dim") or getattr(data_med_config, "dim", None)
    if med_dim_value is None:
        raise ValueError(
            f"Unable to infer med_spec for {experiment_name}. "
            "Provide a med provider with infer_output_spec/infer_output_dim or set data.med.dim."
        )
    return normalize_med_spec(
        {
            "kind": "vector",
            "shape": [int(med_dim_value)],
            "flat_dim": int(med_dim_value),
        }
    )


def build_adam_optimizer(config: Any, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
