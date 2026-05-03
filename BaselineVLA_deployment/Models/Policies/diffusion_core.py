from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Encoders.multimodal_context import MultimodalContextEncoder
from Models.Encoders.task_id_encoder import TaskIDEncoder
from Models.mlp_blocks import build_mlp
from Models.transformer_utils import resolve_transformer_activation

from .policy_support import combine_condition_sequences, format_action_output


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.view(reference.shape[0], *([1] * (reference.dim() - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        device = timesteps.device
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embed_dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ActionDiffusionDenoiser(nn.Module):
    def __init__(self, action_dim: int, action_chunk: int, context_dim: int, config: Any) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        hidden_dim = int(config.model.diffusion_hidden_dim)
        self.action_chunk = int(action_chunk)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, action_chunk, hidden_dim) * 0.02)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(int(config.model.time_embed_dim)),
            nn.Linear(int(config.model.time_embed_dim), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sequence_model: nn.Module | None = None
        if int(config.model.diffusion_transformer_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(config.model.diffusion_num_attention_heads),
                dim_feedforward=int(config.model.diffusion_ffn_dim),
                dropout=float(config.model.dropout),
                activation=resolve_transformer_activation(str(config.model.activation)),
                batch_first=True,
                norm_first=True,
            )
            self.sequence_model = nn.TransformerEncoder(layer, num_layers=int(config.model.diffusion_transformer_layers))
        self.output_head = build_mlp(
            input_dim=hidden_dim,
            hidden_dims=[int(dim) for dim in config.model.diffusion_head_hidden_dims],
            output_dim=action_dim,
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        hidden = self.action_projection(noisy_actions) + self.position_embedding[:, : noisy_actions.shape[1]]
        hidden = hidden + self.context_projection(context_features).unsqueeze(1)
        hidden = hidden + self.time_embedding(timesteps).unsqueeze(1)
        if self.sequence_model is not None:
            hidden = self.sequence_model(hidden)
        return self.output_head(hidden)


class DiffusionPolicyCore(nn.Module):
    """Shared diffusion-style action policy core for multiple experiment paradigms."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, config: Any) -> None:
        super().__init__()
        self.action_chunk = int(action_chunk)
        self.action_dim = int(action_dim)
        self.context_dim = int(config.model.context_dim)
        self.diffusion_steps = int(config.model.diffusion_steps)
        self.inference_steps = int(config.model.inference_steps)
        self.reconstruction_loss_weight = float(config.model.reconstruction_loss_weight)
        task_config = getattr(config.model, "task", None)
        self.use_task_input = bool(getattr(task_config, "enabled", False)) if task_config is not None else False
        self.observation_encoder = MultimodalContextEncoder(
            state_dim=state_dim,
            token_dim=self.context_dim,
            state_hidden_dims=[int(dim) for dim in config.model.state_encoder_hidden_dims],
            vision_token_dim=int(config.model.vision_token_dim),
            vision_hidden_dims=[int(dim) for dim in config.model.vision_encoder_hidden_dims],
            num_attention_heads=int(config.model.context_num_attention_heads),
            num_transformer_layers=int(config.model.context_transformer_layers),
            transformer_ffn_dim=int(config.model.context_ffn_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
            use_vision_tokens=bool(config.model.use_vision_tokens),
        )
        self.task_encoder: TaskIDEncoder | None = None
        if self.use_task_input:
            self.task_encoder = TaskIDEncoder(
                num_embeddings=int(config.model.task.num_embeddings),
                embedding_dim=int(config.model.task.id_embedding_dim),
                output_dim=self.context_dim,
                hidden_dims=[int(dim) for dim in config.model.task.id_encoder_hidden_dims],
                activation=str(config.model.activation),
                dropout=float(config.model.dropout),
            )
        self.denoiser = ActionDiffusionDenoiser(
            action_dim=action_dim,
            action_chunk=self.action_chunk,
            context_dim=self.context_dim,
            config=config,
        )
        betas = torch.linspace(float(config.model.beta_start), float(config.model.beta_end), self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod).clamp_min(1e-8)
        self.register_buffer("posterior_variance", posterior_variance)

    def _prepare_action_sequence(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 2:
            if self.action_chunk != 1:
                raise ValueError(
                    f"Received single-step action target with shape {tuple(action.shape)}, but action_chunk={self.action_chunk}."
                )
            return action.unsqueeze(1)
        if action.dim() == 3:
            if int(action.shape[1]) != self.action_chunk:
                raise ValueError(
                    f"Configured action_chunk={self.action_chunk}, but target chunk={int(action.shape[1])}."
                )
            return action
        raise ValueError(f"Unsupported action shape: {tuple(action.shape)}")

    def _q_sample(self, clean_action: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_alpha_cumprod, timesteps, clean_action) * clean_action + _extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, clean_action
        ) * noise

    def _predict_start_from_noise(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            noisy_action - _extract(self.sqrt_one_minus_alpha_cumprod, timesteps, noisy_action) * predicted_noise
        ) / _extract(self.sqrt_alpha_cumprod, timesteps, noisy_action).clamp_min(1e-6)

    def _p_sample(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.denoiser(noisy_action, timesteps, context_features)
        alpha_t = _extract(self.alphas, timesteps, noisy_action)
        alpha_bar_t = _extract(self.alpha_cumprod, timesteps, noisy_action)
        beta_t = _extract(self.betas, timesteps, noisy_action)
        mean = (noisy_action - (beta_t / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-6)) * predicted_noise) / alpha_t.sqrt().clamp_min(1e-6)
        variance = _extract(self.posterior_variance, timesteps, noisy_action)
        nonzero_mask = (timesteps > 0).float().view(noisy_action.shape[0], *([1] * (noisy_action.dim() - 1)))
        noise = torch.randn_like(noisy_action)
        return mean + nonzero_mask * variance.sqrt() * noise

    def resolve_task_tokens(
        self,
        state: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.use_task_input and self.task_encoder is not None and task_tokens is None:
            if task_id is None:
                raise ValueError("Configured task-conditioned diffusion policy expects task_id in model inputs.")
            encoded_task = self.task_encoder(task_id.to(device=state.device, dtype=torch.long))
            task_tokens = encoded_task["tokens"]
            task_mask = encoded_task["mask"]
        return task_tokens, task_mask

    def encode_context(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        condition_tokens: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        task_tokens, task_mask = self.resolve_task_tokens(
            state=state,
            task_id=task_id,
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

    def sample_action_sequence(self, context_features: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        action_sequence = torch.randn(
            context_features.shape[0],
            self.action_chunk,
            self.action_dim,
            dtype=dtype,
            device=device,
        )
        sample_steps = min(self.inference_steps, self.diffusion_steps)
        sample_schedule = torch.linspace(self.diffusion_steps - 1, 0, steps=sample_steps, device=device)
        for timestep_float in sample_schedule:
            timestep = torch.full((context_features.shape[0],), int(timestep_float.item()), device=device, dtype=torch.long)
            action_sequence = self._p_sample(action_sequence, timestep, context_features)
        return action_sequence

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
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
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=condition_tokens,
            condition_mask=condition_mask,
        )
        action_sequence = self.sample_action_sequence(encoded["pooled"], dtype=state.dtype, device=state.device)
        return {
            "action": format_action_output(action_sequence),
            "action_sequence": action_sequence,
            "context_features": encoded["pooled"],
        }

    def compute_loss_from_context(
        self,
        context_features: torch.Tensor,
        target_action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        action_sequence = self._prepare_action_sequence(target_action)
        noise = torch.randn_like(action_sequence)
        timesteps = torch.randint(0, self.diffusion_steps, (action_sequence.shape[0],), device=action_sequence.device).long()
        noisy_action = self._q_sample(action_sequence, noise, timesteps)
        predicted_noise = self.denoiser(noisy_action, timesteps, context_features)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        reconstructed_action = self._predict_start_from_noise(noisy_action, timesteps, predicted_noise)
        formatted_action = format_action_output(reconstructed_action)
        reconstruction_loss = F.mse_loss(formatted_action, target_action)
        total_loss = diffusion_loss + self.reconstruction_loss_weight * reconstruction_loss
        outputs = {
            "action": formatted_action,
            "action_sequence": reconstructed_action,
            "context_features": context_features,
        }
        component_losses = {
            "diffusion": diffusion_loss,
            "action": reconstruction_loss,
            "action_recon": reconstruction_loss,
        }
        return outputs, total_loss, component_losses

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        encoded = self.encode_context(
            state=model_inputs["state"],
            vision_tokens=model_inputs.get("vision_tokens"),
            vision_mask=model_inputs.get("vision_mask"),
            task_id=model_inputs.get("task_id"),
            task_tokens=model_inputs.get("task_tokens"),
            task_mask=model_inputs.get("task_mask"),
            condition_tokens=model_inputs.get("condition_tokens"),
            condition_mask=model_inputs.get("condition_mask"),
        )
        return self.compute_loss_from_context(context_features=encoded["pooled"], target_action=targets["action"])
