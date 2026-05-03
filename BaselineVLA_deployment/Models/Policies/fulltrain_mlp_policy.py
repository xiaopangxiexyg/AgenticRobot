from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.med_support import build_med_predictor, build_med_projector
from Models.Encoders.state_mlp import StateMLPEncoder
from Models.Encoders.task_id_encoder import TaskIDEncoder
from Models.Policies.action_mlp import ActionMLPHead

from .policy_support import format_action_output


class FullTrainMLPPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, med_spec: Dict[str, Any], action_chunk: int, config: Any) -> None:
        super().__init__()
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.state_encoder = StateMLPEncoder(
            input_dim=state_dim,
            hidden_dims=[int(dim) for dim in config.model.state_encoder_hidden_dims],
            feature_dim=int(config.model.state_feature_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )
        self.task_encoder: TaskIDEncoder | None = None
        state_feature_dim = int(config.model.state_feature_dim)
        med_predictor_input_dim = state_feature_dim
        action_input_dim = state_feature_dim + int(config.model.med_feature_dim)
        if self.use_task_input:
            self.task_encoder = TaskIDEncoder(
                num_embeddings=int(config.model.task.num_embeddings),
                embedding_dim=int(config.model.task.id_embedding_dim),
                output_dim=int(config.model.task.feature_dim),
                hidden_dims=[int(dim) for dim in config.model.task.id_encoder_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
            med_predictor_input_dim += int(config.model.task.feature_dim)
            action_input_dim += int(config.model.task.feature_dim)
        self.med_predictor = build_med_predictor(config=config, med_spec=med_spec, input_dim=med_predictor_input_dim)
        self.med_projector = build_med_projector(config=config, med_spec=med_spec, feature_dim=int(config.model.med_feature_dim))
        self.action_head = ActionMLPHead(
            input_dim=action_input_dim,
            hidden_dims=[int(dim) for dim in config.model.action_head_hidden_dims],
            action_dim=action_dim * self.action_chunk,
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )

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
        feature_chunks = [state_features]
        if self.use_task_input and self.task_encoder is not None:
            if task_id is None:
                raise ValueError("Configured task-conditioned FullTrain MLP expects task_id in model inputs.")
            feature_chunks.append(self.task_encoder(task_id)["features"])
        fused_state_features = torch.cat(feature_chunks, dim=-1)
        med_prediction = self.med_predictor(fused_state_features)
        med_features = self.med_projector(med_prediction)
        fused_features = torch.cat([fused_state_features, med_features], dim=-1)
        action_sequence = self.action_head(fused_features).view(state.shape[0], self.action_chunk, self.action_dim)
        return {
            "med": med_prediction,
            "action": format_action_output(action_sequence),
            "action_sequence": action_sequence,
        }
