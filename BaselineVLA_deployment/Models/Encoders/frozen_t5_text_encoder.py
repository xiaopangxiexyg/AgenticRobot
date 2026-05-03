from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp
from utils.path_config import resolve_repo_path


class FrozenT5TaskEncoder(nn.Module):
    """Frozen local T5 text encoder used to convert task descriptions into task tokens."""

    def __init__(
        self,
        model_path: str,
        output_dim: int,
        max_length: int,
        projection_hidden_dims: list[int] | None = None,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as error:
            raise ImportError(
                "Task-conditioned VLA requires the 'transformers' package. "
                "Install transformers and sentencepiece in the ManipRep_310 environment before enabling model.task.enabled=true."
            ) from error

        resolved_path = resolve_repo_path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configured T5 path does not exist: {resolved_path}")

        self.max_length = int(max_length)
        self.tokenizer = T5Tokenizer.from_pretrained(str(resolved_path), local_files_only=True)
        self.text_encoder = T5EncoderModel.from_pretrained(str(resolved_path), local_files_only=True)
        self.text_encoder.eval()
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        self.projector = build_mlp(
            input_dim=int(self.text_encoder.config.d_model),
            hidden_dims=[] if projection_hidden_dims is None else projection_hidden_dims,
            output_dim=int(output_dim),
            activation=activation,
            dropout=dropout,
        )

    def forward(self, task_text: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
        normalized_text = [str(text) if str(text).strip() else "unknown_task" for text in task_text]
        encoded_inputs = self.tokenizer(
            normalized_text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(**encoded_inputs)
            hidden_states = outputs.last_hidden_state

        task_tokens = self.projector(hidden_states)
        task_mask = encoded_inputs["attention_mask"].to(dtype=torch.bool)
        return {
            "tokens": task_tokens,
            "mask": task_mask,
        }
