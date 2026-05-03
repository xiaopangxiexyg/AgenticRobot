from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.med_support import flatten_med_to_sequence, normalize_med_spec, restore_sequence_to_med

from .diffusion_core import DiffusionPolicyCore


class GenMedDiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int, med_spec: dict[str, Any], config: Any) -> None:
        super().__init__()
        self.med_spec = normalize_med_spec(med_spec)
        med_shape = list(self.med_spec["shape"])
        if len(med_shape) <= 1:
            sequence_length = 1
            feature_dim = int(self.med_spec["flat_dim"])
        else:
            sequence_length = 1
            for value in med_shape[:-1]:
                sequence_length *= int(value)
            feature_dim = int(med_shape[-1])
        self.core = DiffusionPolicyCore(
            state_dim=state_dim,
            action_dim=int(feature_dim),
            action_chunk=int(sequence_length),
            config=config,
        )

    def _restore_med(self, action_like_sequence: torch.Tensor) -> torch.Tensor:
        if action_like_sequence.dim() == 2:
            action_like_sequence = action_like_sequence.unsqueeze(1)
        return restore_sequence_to_med(action_like_sequence, self.med_spec)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        outputs = self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        med_prediction = self._restore_med(outputs["action_sequence"])
        return {
            "med": med_prediction,
            "med_sequence": flatten_med_to_sequence(med_prediction, self.med_spec),
            "context_features": outputs["context_features"],
        }

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        encoded = self.core.encode_context(
            state=model_inputs["state"],
            vision_tokens=model_inputs.get("vision_tokens"),
            vision_mask=model_inputs.get("vision_mask"),
            task_id=model_inputs.get("task_id"),
            task_tokens=model_inputs.get("task_tokens"),
            task_mask=model_inputs.get("task_mask"),
        )
        target_sequence = flatten_med_to_sequence(targets["med"], self.med_spec)
        if int(self.core.action_chunk) == 1:
            target_sequence = target_sequence[:, 0]
        outputs, total_loss, component_losses = self.core.compute_loss_from_context(
            context_features=encoded["pooled"],
            target_action=target_sequence,
        )
        med_prediction = self._restore_med(outputs["action_sequence"])
        renamed_component_losses = {
            "diffusion": component_losses["diffusion"],
            "med": component_losses["action"],
            "med_recon": component_losses["action_recon"],
        }
        return {
            "med": med_prediction,
            "med_sequence": flatten_med_to_sequence(med_prediction, self.med_spec),
            "context_features": outputs["context_features"],
        }, total_loss, renamed_component_losses
