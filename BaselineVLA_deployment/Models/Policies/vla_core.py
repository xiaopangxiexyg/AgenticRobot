from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.Encoders.frozen_t5_text_encoder import FrozenT5TaskEncoder
from Models.Encoders.multimodal_context import MultimodalContextEncoder
from Models.mlp_blocks import build_mlp
from Models.transformer_utils import resolve_transformer_activation

from .policy_support import combine_condition_sequences, format_action_output


class VLAPolicyCore(nn.Module):
    """Shared VLA-style token policy core for multiple experiment paradigms."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, config: Any) -> None:
        super().__init__()
        token_dim = int(config.model.token_dim)
        self.token_dim = token_dim
        self.action_dim = int(action_dim)
        self.action_chunk = int(action_chunk)
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.observation_encoder = MultimodalContextEncoder(
            state_dim=state_dim,
            token_dim=token_dim,
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
                output_dim=token_dim,
                max_length=int(config.model.task.max_text_tokens),
                projection_hidden_dims=[int(dim) for dim in config.model.task.text_projection_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
        self.action_queries = nn.Parameter(torch.randn(1, self.action_chunk, token_dim) * 0.02)
        self.action_positions = nn.Parameter(torch.randn(1, self.action_chunk, token_dim) * 0.02)
        self.policy_transformer: nn.Module | None = None
        if int(config.model.policy_transformer_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=int(config.model.num_attention_heads),
                dim_feedforward=int(config.model.transformer_ffn_dim),
                dropout=float(config.model.dropout),
                activation=resolve_transformer_activation(str(config.model.activation)),
                batch_first=True,
                norm_first=True,
            )
            self.policy_transformer = nn.TransformerEncoder(layer, num_layers=int(config.model.policy_transformer_layers))
        self.query_norm = nn.LayerNorm(token_dim)
        self.action_head = build_mlp(
            input_dim=token_dim,
            hidden_dims=[int(dim) for dim in config.model.action_decoder_hidden_dims],
            output_dim=int(action_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )

    def resolve_task_tokens(
        self,
        state: torch.Tensor,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.use_task_input and self.task_encoder is not None and task_tokens is None:
            if task_text is None:
                raise ValueError("Configured task-conditioned VLA policy expects task_text in model inputs.")
            encoded_task = self.task_encoder(task_text=task_text, device=state.device)
            task_tokens = encoded_task["tokens"]
            task_mask = encoded_task["mask"]
        return task_tokens, task_mask

    def encode_context(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        task_tokens, task_mask = self.resolve_task_tokens(
            state=state,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        extra_tokens, extra_mask = combine_condition_sequences(
            (task_tokens, task_mask),
            (condition_tokens, condition_mask),
        )
        return self.observation_encoder(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            extra_tokens=extra_tokens,
            extra_mask=extra_mask,
        )

    def decode_action(self, encoded_context: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = int(encoded_context["tokens"].shape[0])
        device = encoded_context["tokens"].device
        query_tokens = self.action_queries.expand(batch_size, -1, -1) + self.action_positions
        query_mask = torch.ones(batch_size, self.action_chunk, dtype=torch.bool, device=device)
        tokens = torch.cat([query_tokens, encoded_context["tokens"]], dim=1)
        mask = torch.cat([query_mask, encoded_context["mask"]], dim=1)
        if self.policy_transformer is not None:
            tokens = self.policy_transformer(tokens, src_key_padding_mask=~mask)
        action_tokens = self.query_norm(tokens[:, : self.action_chunk])
        return self.action_head(action_tokens)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode_context(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
        )
        action_sequence = self.decode_action(encoded)
        return {
            "action": format_action_output(action_sequence),
            "action_sequence": action_sequence,
            "context_features": encoded["pooled"],
        }
