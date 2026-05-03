from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.Encoders.frozen_t5_text_encoder import FrozenT5TaskEncoder
from Models.Encoders.multimodal_context import MultimodalContextEncoder
from Models.med_support import build_med_predictor


class GenMedVLAPolicy(nn.Module):
    def __init__(self, state_dim: int, med_spec: dict[str, Any], config: Any) -> None:
        super().__init__()
        self.token_dim = int(config.model.token_dim)
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.observation_encoder = MultimodalContextEncoder(
            state_dim=state_dim,
            token_dim=self.token_dim,
            state_hidden_dims=[int(dim) for dim in config.model.state_encoder_hidden_dims],
            vision_token_dim=int(config.model.vision_token_dim),
            vision_hidden_dims=[int(dim) for dim in config.model.vision_encoder_hidden_dims],
            num_attention_heads=int(config.model.num_attention_heads),
            num_transformer_layers=int(config.model.observation_transformer_layers),
            transformer_ffn_dim=int(config.model.transformer_ffn_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
            use_vision_tokens=bool(config.model.use_vision_tokens),
        )
        self.task_encoder: FrozenT5TaskEncoder | None = None
        if self.use_task_input:
            self.task_encoder = FrozenT5TaskEncoder(
                model_path=str(config.model.task.t5_model_path),
                output_dim=self.token_dim,
                max_length=int(config.model.task.max_text_tokens),
                projection_hidden_dims=[int(dim) for dim in config.model.task.text_projection_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
        self.med_predictor = build_med_predictor(config=config, med_spec=med_spec, input_dim=self.token_dim)

    def _resolve_task_tokens(
        self,
        state: torch.Tensor,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.use_task_input and self.task_encoder is not None and task_tokens is None:
            if task_text is None:
                raise ValueError("Configured task-conditioned GenMed VLA expects task_text in model inputs.")
            encoded_task = self.task_encoder(task_text=task_text, device=state.device)
            task_tokens = encoded_task["tokens"]
            task_mask = encoded_task["mask"]
        return task_tokens, task_mask

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        task_tokens, task_mask = self._resolve_task_tokens(
            state=state,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        encoded = self.observation_encoder(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            extra_tokens=task_tokens,
            extra_mask=task_mask,
        )
        med_prediction = self.med_predictor(encoded["pooled"])
        return {
            "med": med_prediction,
            "context_features": encoded["pooled"],
        }
