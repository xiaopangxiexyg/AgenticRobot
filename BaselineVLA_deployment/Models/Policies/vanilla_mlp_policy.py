from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.Encoders.state_mlp import StateMLPEncoder
from Models.Encoders.task_id_encoder import TaskIDEncoder
from Models.Policies.action_mlp import ActionMLPHead


def _format_action_output(action_sequence: torch.Tensor) -> torch.Tensor:
    if int(action_sequence.shape[1]) == 1:
        return action_sequence[:, 0]
    return action_sequence


class BasicVanillaMLPPolicy(nn.Module):
    """Baseline low-dimensional policy that only consumes state."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, config: Any) -> None:
        super().__init__()
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.action_chunk = int(action_chunk)
        self.state_encoder = StateMLPEncoder(
            input_dim=state_dim,
            hidden_dims=[int(dim) for dim in config.model.state_encoder_hidden_dims],
            feature_dim=int(config.model.state_feature_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )
        self.task_encoder: TaskIDEncoder | None = None
        action_input_dim = int(config.model.state_feature_dim)
        if self.use_task_input:
            self.task_encoder = TaskIDEncoder(
                num_embeddings=int(config.model.task.num_embeddings),
                embedding_dim=int(config.model.task.id_embedding_dim),
                output_dim=int(config.model.task.feature_dim),
                hidden_dims=[int(dim) for dim in config.model.task.id_encoder_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
            action_input_dim += int(config.model.task.feature_dim)
        self.action_head = ActionMLPHead(
            input_dim=action_input_dim,
            hidden_dims=[int(dim) for dim in config.model.action_head_hidden_dims],
            action_dim=action_dim * self.action_chunk,
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )
        self.action_dim = int(action_dim)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        del vision_tokens, vision_mask
        state_features = self.state_encoder(state)
        if self.use_task_input and self.task_encoder is not None:
            if task_id is None:
                raise ValueError("Configured task-conditioned MLP policy expects task_id in model inputs.")
            task_features = self.task_encoder(task_id)["features"]
            state_features = torch.cat([state_features, task_features], dim=-1)
        action_sequence = self.action_head(state_features).view(state.shape[0], self.action_chunk, self.action_dim)
        return {"action": _format_action_output(action_sequence), "action_sequence": action_sequence}
