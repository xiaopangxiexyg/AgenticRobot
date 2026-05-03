from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.Encoders.state_mlp import StateMLPEncoder
from Models.Encoders.task_id_encoder import TaskIDEncoder
from Models.med_support import build_med_predictor


class GenMedMLPPolicy(nn.Module):
    def __init__(self, state_dim: int, med_spec: Dict[str, Any], config: Any) -> None:
        super().__init__()
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.state_encoder = StateMLPEncoder(
            input_dim=state_dim,
            hidden_dims=[int(dim) for dim in config.model.state_encoder_hidden_dims],
            feature_dim=int(config.model.state_feature_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )
        predictor_input_dim = int(config.model.state_feature_dim)
        self.task_encoder: TaskIDEncoder | None = None
        if self.use_task_input:
            self.task_encoder = TaskIDEncoder(
                num_embeddings=int(config.model.task.num_embeddings),
                embedding_dim=int(config.model.task.id_embedding_dim),
                output_dim=int(config.model.task.feature_dim),
                hidden_dims=[int(dim) for dim in config.model.task.id_encoder_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
            predictor_input_dim += int(config.model.task.feature_dim)
        self.med_predictor = build_med_predictor(config=config, med_spec=med_spec, input_dim=predictor_input_dim)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        task_id: torch.Tensor | None = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        del vision_tokens, vision_mask
        feature_chunks = [self.state_encoder(state)]
        if self.use_task_input and self.task_encoder is not None:
            if task_id is None:
                raise ValueError("Configured task-conditioned GenMed MLP expects task_id in model inputs.")
            feature_chunks.append(self.task_encoder(task_id)["features"])
        fused_features = torch.cat(feature_chunks, dim=-1)
        med_prediction = self.med_predictor(fused_features)
        return {
            "med": med_prediction,
            "context_features": fused_features,
        }
