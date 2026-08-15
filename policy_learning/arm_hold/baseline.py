"""Small independent arm-hold learner and its single canonical loss.

This is intentionally not an OpenPI adapter.  It provides a cheap reference
model for validating input/label plumbing before the same frozen dataset view
is connected to a larger policy framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset_contracts.arm_hold_history import (
    ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1,
    ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1,
    ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1,
)
from dataset_interface.arm_hold_windows import ARM_HOLD_INPUT_MODES, ArmHoldWindowSample


ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1 = (
    ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
)


ARM_HOLD_LABEL_CONTEXT_MODES = frozenset(
    {
        "none",
        "previous_mode",
        "previous_dwell_bucket",
        "previous_mode_dwell",
        "previous_two_dwell_bucket",
        "previous_two_mode_dwell",
    }
)
ARM_HOLD_FUSION_MODES = frozenset(
    {"concat_v1", "state_history_visual_residual_v1"}
)
ARM_HOLD_DWELL_BUCKET_THRESHOLDS = ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1


class ArmHoldBaselineError(ValueError):
    """Raised when a baseline batch/config violates the frozen semantics."""


def arm_hold_dwell_bucket_indices(
    dwell_steps: torch.Tensor,
    available_mask: torch.Tensor,
) -> torch.Tensor:
    """Map unavailable/1/2/3/4-7/8-15/16+ to categorical indices 0..6."""

    if (
        dwell_steps.shape != available_mask.shape
        or dwell_steps.dtype != torch.int64
        or available_mask.dtype != torch.bool
        or bool(torch.any(dwell_steps < 0))
        or not torch.equal(dwell_steps > 0, available_mask)
    ):
        raise ArmHoldBaselineError("History dwell bucket input is invalid.")
    bucket = available_mask.to(dtype=torch.int64)
    for threshold in ARM_HOLD_DWELL_BUCKET_THRESHOLDS:
        bucket = bucket + (dwell_steps >= threshold).to(dtype=torch.int64)
    return bucket


@dataclass(frozen=True)
class ArmHoldBaselineConfig:
    input_mode: str
    state_dim: int
    horizon: int
    image_feature_dim: int = 64
    state_feature_dim: int = 32
    fusion_dim: int = 96
    label_context_mode: str = "none"
    label_context_feature_dim: int = 8
    fusion_mode: str = "concat_v1"

    def __post_init__(self) -> None:
        mode = str(self.input_mode).strip().lower()
        context_mode = str(self.label_context_mode).strip().lower()
        fusion_mode = str(self.fusion_mode).strip().lower()
        if mode not in ARM_HOLD_INPUT_MODES:
            raise ArmHoldBaselineError(f"Unsupported input_mode {self.input_mode!r}.")
        if context_mode not in ARM_HOLD_LABEL_CONTEXT_MODES:
            raise ArmHoldBaselineError(
                f"Unsupported label_context_mode {self.label_context_mode!r}."
            )
        if fusion_mode not in ARM_HOLD_FUSION_MODES:
            raise ArmHoldBaselineError(
                f"Unsupported fusion_mode {self.fusion_mode!r}."
            )
        if (
            fusion_mode == "state_history_visual_residual_v1"
            and mode != "front_state"
        ):
            raise ArmHoldBaselineError(
                "state_history_visual_residual_v1 requires front_state input."
            )
        for name in ("horizon", "image_feature_dim", "state_feature_dim", "fusion_dim"):
            if int(getattr(self, name)) <= 0:
                raise ArmHoldBaselineError(f"{name} must be positive.")
        if mode in {"front_state", "state_only"} and int(self.state_dim) <= 0:
            raise ArmHoldBaselineError("state_dim must be positive when state is used.")
        if mode == "front_only" and int(self.state_dim) < 0:
            raise ArmHoldBaselineError("state_dim cannot be negative.")
        if context_mode != "none" and int(self.label_context_feature_dim) <= 0:
            raise ArmHoldBaselineError(
                "label_context_feature_dim must be positive when context is used."
            )
        object.__setattr__(self, "input_mode", mode)
        object.__setattr__(self, "label_context_mode", context_mode)
        object.__setattr__(self, "fusion_mode", fusion_mode)


@dataclass(frozen=True)
class ArmHoldLossConfig:
    positive_weight_left: float = 1.0
    positive_weight_right: float = 1.0
    exact_transition_weight: float = 1.0
    stable_persistence_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "positive_weight_left",
            "positive_weight_right",
            "exact_transition_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ArmHoldBaselineError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        persistence = float(self.stable_persistence_weight)
        if not math.isfinite(persistence) or persistence < 0.0:
            raise ArmHoldBaselineError(
                "stable_persistence_weight must be finite and nonnegative."
            )
        object.__setattr__(self, "stable_persistence_weight", persistence)


@dataclass(frozen=True)
class ArmHoldTensorBatch:
    front_rgb: torch.Tensor | None
    observation_state: torch.Tensor | None
    target: torch.Tensor
    valid_mask: torch.Tensor
    loss_mask: torch.Tensor
    transition_mask: torch.Tensor
    metadata: tuple[dict[str, Any], ...]
    history_previous_mode: torch.Tensor | None = None
    history_available_mask: torch.Tensor | None = None
    history_dwell_steps: torch.Tensor | None = None
    history_sequence_previous_mode: torch.Tensor | None = None
    history_sequence_available_mask: torch.Tensor | None = None
    history_sequence_dwell_steps: torch.Tensor | None = None
    history_context_provenance: str | None = None

    def __post_init__(self) -> None:
        target = self.target
        if target.ndim != 3 or target.shape[-1] != 2:
            raise ArmHoldBaselineError("target must have shape [B, H, 2].")
        for name in ("valid_mask", "loss_mask", "transition_mask"):
            value = getattr(self, name)
            if value.shape != target.shape or value.dtype != torch.bool:
                raise ArmHoldBaselineError(
                    f"{name} must be bool with shape {tuple(target.shape)}."
                )
        if target.dtype != torch.float32:
            raise ArmHoldBaselineError("target must be float32.")
        batch = target.shape[0]
        if len(self.metadata) != batch:
            raise ArmHoldBaselineError("metadata count must match batch size.")
        one_step_values = (
            self.history_previous_mode,
            self.history_available_mask,
            self.history_dwell_steps,
        )
        sequence_values = (
            self.history_sequence_previous_mode,
            self.history_sequence_available_mask,
            self.history_sequence_dwell_steps,
        )
        has_context = one_step_values[0] is not None
        has_sequence = sequence_values[0] is not None
        if any((value is not None) != has_context for value in one_step_values):
            raise ArmHoldBaselineError("History context tensors must be all-or-none.")
        if any((value is not None) != has_sequence for value in sequence_values):
            raise ArmHoldBaselineError(
                "History sequence tensors must be all-or-none."
            )
        if has_context and has_sequence:
            raise ArmHoldBaselineError(
                "One-step and two-step history tensors are mutually exclusive."
            )
        if has_context:
            previous = self.history_previous_mode
            available = self.history_available_mask
            dwell = self.history_dwell_steps
            assert previous is not None and available is not None and dwell is not None
            if (
                previous.shape != (batch, 2)
                or previous.dtype != torch.float32
                or available.shape != (batch, 2)
                or available.dtype != torch.bool
                or dwell.shape != (batch, 2)
                or dwell.dtype != torch.int64
            ):
                raise ArmHoldBaselineError("History context tensor contract is invalid.")
            if bool(torch.any(dwell < 0)) or not torch.equal(dwell > 0, available):
                raise ArmHoldBaselineError(
                    "History dwell must be positive exactly where context is available."
                )
            available_previous = previous[available]
            invalid_available = (available_previous != 0.0) & (
                available_previous != 1.0
            )
            if bool(torch.any(invalid_available)) or bool(
                torch.any(previous[~available] != 0.0)
            ):
                raise ArmHoldBaselineError("History previous modes are invalid.")
            if self.history_context_provenance not in (
                ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1
            ):
                raise ArmHoldBaselineError("History context provenance is invalid.")
        elif has_sequence:
            previous = self.history_sequence_previous_mode
            available = self.history_sequence_available_mask
            dwell = self.history_sequence_dwell_steps
            assert previous is not None and available is not None and dwell is not None
            if (
                previous.shape != (batch, 2, 2)
                or previous.dtype != torch.float32
                or available.shape != (batch, 2, 2)
                or available.dtype != torch.bool
                or dwell.shape != (batch, 2, 2)
                or dwell.dtype != torch.int64
            ):
                raise ArmHoldBaselineError(
                    "History sequence tensor contract is invalid."
                )
            if bool(torch.any(dwell < 0)) or not torch.equal(dwell > 0, available):
                raise ArmHoldBaselineError(
                    "History sequence dwell availability is invalid."
                )
            available_previous = previous[available]
            invalid_available = (available_previous != 0.0) & (
                available_previous != 1.0
            )
            if bool(torch.any(invalid_available)) or bool(
                torch.any(previous[~available] != 0.0)
            ):
                raise ArmHoldBaselineError("History sequence modes are invalid.")
            if bool(torch.any(available[:, 0] & ~available[:, 1])):
                raise ArmHoldBaselineError(
                    "History sequence availability must be a suffix."
                )
            both = available[:, 0] & available[:, 1]
            same = previous[:, 0] == previous[:, 1]
            if bool(
                torch.any(both & same & (dwell[:, 1] != dwell[:, 0] + 1))
                or torch.any(both & ~same & (dwell[:, 1] != 1))
            ):
                raise ArmHoldBaselineError(
                    "History sequence mode/dwell progression is invalid."
                )
            if self.history_context_provenance not in (
                ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1
            ):
                raise ArmHoldBaselineError("History context provenance is invalid.")
        elif self.history_context_provenance is not None:
            raise ArmHoldBaselineError(
                "History context provenance requires context tensors."
            )
        if self.front_rgb is not None and (
            self.front_rgb.dtype != torch.float32
            or self.front_rgb.ndim != 4
            or self.front_rgb.shape[0] != batch
            or self.front_rgb.shape[1] != 3
        ):
            raise ArmHoldBaselineError("front_rgb must be float32 [B,3,H,W].")
        if self.observation_state is not None and (
            self.observation_state.dtype != torch.float32
            or self.observation_state.ndim != 2
            or self.observation_state.shape[0] != batch
        ):
            raise ArmHoldBaselineError(
                "observation_state must be float32 [B,state_dim]."
            )

    def to(self, device: torch.device | str) -> "ArmHoldTensorBatch":
        return ArmHoldTensorBatch(
            front_rgb=(
                None if self.front_rgb is None else self.front_rgb.to(device)
            ),
            observation_state=(
                None
                if self.observation_state is None
                else self.observation_state.to(device)
            ),
            target=self.target.to(device),
            valid_mask=self.valid_mask.to(device),
            loss_mask=self.loss_mask.to(device),
            transition_mask=self.transition_mask.to(device),
            metadata=self.metadata,
            history_previous_mode=(
                None
                if self.history_previous_mode is None
                else self.history_previous_mode.to(device)
            ),
            history_available_mask=(
                None
                if self.history_available_mask is None
                else self.history_available_mask.to(device)
            ),
            history_dwell_steps=(
                None
                if self.history_dwell_steps is None
                else self.history_dwell_steps.to(device)
            ),
            history_sequence_previous_mode=(
                None
                if self.history_sequence_previous_mode is None
                else self.history_sequence_previous_mode.to(device)
            ),
            history_sequence_available_mask=(
                None
                if self.history_sequence_available_mask is None
                else self.history_sequence_available_mask.to(device)
            ),
            history_sequence_dwell_steps=(
                None
                if self.history_sequence_dwell_steps is None
                else self.history_sequence_dwell_steps.to(device)
            ),
            history_context_provenance=self.history_context_provenance,
        )


def arm_hold_tensor_batch(
    samples: Sequence[ArmHoldWindowSample],
) -> ArmHoldTensorBatch:
    """Collate framework-neutral samples without augmentation or resizing."""

    values = tuple(samples)
    if not values:
        raise ArmHoldBaselineError("Cannot collate an empty batch.")
    has_front = values[0].front_rgb is not None
    has_state = values[0].observation_state is not None
    has_context = values[0].arm_hold_history_context is not None
    has_sequence = values[0].arm_hold_history_sequence_context is not None
    if any((item.front_rgb is not None) != has_front for item in values):
        raise ArmHoldBaselineError("Batch mixes samples with and without front RGB.")
    if any((item.observation_state is not None) != has_state for item in values):
        raise ArmHoldBaselineError("Batch mixes samples with and without state.")
    if any(
        (item.arm_hold_history_context is not None) != has_context for item in values
    ):
        raise ArmHoldBaselineError(
            "Batch mixes samples with and without history context."
        )
    if any(
        (item.arm_hold_history_sequence_context is not None) != has_sequence
        for item in values
    ):
        raise ArmHoldBaselineError(
            "Batch mixes samples with and without history sequence context."
        )
    if has_context and has_sequence:
        raise ArmHoldBaselineError(
            "Batch cannot carry one-step and two-step history together."
        )
    front = None
    if has_front:
        images = np.stack([item.front_rgb for item in values], axis=0)
        front = (
            torch.from_numpy(np.ascontiguousarray(images))
            .permute(0, 3, 1, 2)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
    state = None
    if has_state:
        state = torch.from_numpy(
            np.stack([item.observation_state for item in values], axis=0)
        ).to(dtype=torch.float32)
    history_previous_mode = None
    history_available_mask = None
    history_dwell_steps = None
    history_sequence_previous_mode = None
    history_sequence_available_mask = None
    history_sequence_dwell_steps = None
    history_context_provenance = None
    if has_context:
        contexts = tuple(item.arm_hold_history_context for item in values)
        assert all(item is not None for item in contexts)
        resolved_contexts = tuple(item for item in contexts if item is not None)
        provenances = {item.provenance for item in resolved_contexts}
        if provenances != {ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1}:
            raise ArmHoldBaselineError("Batch history context provenance differs.")
        history_previous_mode = torch.from_numpy(
            np.stack([item.previous_mode for item in resolved_contexts], axis=0)
        ).to(dtype=torch.float32)
        history_available_mask = torch.from_numpy(
            np.stack([item.available_mask for item in resolved_contexts], axis=0)
        ).to(dtype=torch.bool)
        history_dwell_steps = torch.from_numpy(
            np.stack([item.dwell_steps for item in resolved_contexts], axis=0)
        ).to(dtype=torch.int64)
        history_context_provenance = next(iter(provenances))
    elif has_sequence:
        contexts = tuple(item.arm_hold_history_sequence_context for item in values)
        assert all(item is not None for item in contexts)
        resolved_contexts = tuple(item for item in contexts if item is not None)
        provenances = {item.provenance for item in resolved_contexts}
        if provenances != {ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1}:
            raise ArmHoldBaselineError(
                "Batch history sequence provenance differs."
            )
        history_sequence_previous_mode = torch.from_numpy(
            np.stack([item.previous_mode for item in resolved_contexts], axis=0)
        ).to(dtype=torch.float32)
        history_sequence_available_mask = torch.from_numpy(
            np.stack([item.available_mask for item in resolved_contexts], axis=0)
        ).to(dtype=torch.bool)
        history_sequence_dwell_steps = torch.from_numpy(
            np.stack([item.dwell_steps for item in resolved_contexts], axis=0)
        ).to(dtype=torch.int64)
        history_context_provenance = next(iter(provenances))
    return ArmHoldTensorBatch(
        front_rgb=front,
        observation_state=state,
        target=torch.from_numpy(
            np.stack([item.arm_hold_target for item in values], axis=0)
        ).to(dtype=torch.float32),
        valid_mask=torch.from_numpy(
            np.stack([item.arm_hold_valid_mask for item in values], axis=0)
        ).to(dtype=torch.bool),
        loss_mask=torch.from_numpy(
            np.stack([item.arm_hold_loss_mask for item in values], axis=0)
        ).to(dtype=torch.bool),
        transition_mask=torch.from_numpy(
            np.stack([item.arm_hold_transition_mask for item in values], axis=0)
        ).to(dtype=torch.bool),
        metadata=tuple(dict(item.metadata) for item in values),
        history_previous_mode=history_previous_mode,
        history_available_mask=history_available_mask,
        history_dwell_steps=history_dwell_steps,
        history_sequence_previous_mode=history_sequence_previous_mode,
        history_sequence_available_mask=history_sequence_available_mask,
        history_sequence_dwell_steps=history_sequence_dwell_steps,
        history_context_provenance=history_context_provenance,
    )


def _arm_hold_image_encoder(config: ArmHoldBaselineConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, config.image_feature_dim),
        nn.ReLU(inplace=True),
    )


class ArmHoldBaseline(nn.Module):
    """Current RGB/state fusion with one parallel horizon head."""

    def __init__(self, config: ArmHoldBaselineConfig) -> None:
        super().__init__()
        self.config = config
        use_front = config.input_mode in {"front_state", "front_only"}
        use_state = config.input_mode in {"front_state", "state_only"}
        visual_residual = (
            config.fusion_mode == "state_history_visual_residual_v1"
        )
        self.image_encoder: nn.Module | None = None
        self.state_encoder: nn.Module | None = None
        self.history_mode_embedding: nn.Embedding | None = None
        self.history_dwell_embedding: nn.Embedding | None = None
        self.history_sequence_mode_embedding: nn.Embedding | None = None
        self.history_sequence_dwell_embedding: nn.Embedding | None = None
        self.visual_residual_head: nn.Module | None = None
        feature_dim = 0
        if use_front and not visual_residual:
            self.image_encoder = _arm_hold_image_encoder(config)
            feature_dim += config.image_feature_dim
        if use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.state_dim, config.state_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.state_feature_dim, config.state_feature_dim),
                nn.ReLU(inplace=True),
            )
            feature_dim += config.state_feature_dim
        if config.label_context_mode in {"previous_mode", "previous_mode_dwell"}:
            self.history_mode_embedding = nn.Embedding(
                3, config.label_context_feature_dim
            )
        if config.label_context_mode != "none":
            context_steps = (
                2 if config.label_context_mode.startswith("previous_two_") else 1
            )
            feature_dim += 2 * context_steps * config.label_context_feature_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, config.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.fusion_dim, config.horizon * 2),
        )
        # Construct after the shared head so previous_mode and
        # previous_mode_dwell use identical shared initialization for a seed.
        if config.label_context_mode in {
            "previous_dwell_bucket",
            "previous_mode_dwell",
        }:
            self.history_dwell_embedding = nn.Embedding(
                7, config.label_context_feature_dim
            )
        # H2 variants construct their shared head and dwell encoder before the
        # optional mode encoder, so dwell-only and mode+dwell share identical
        # common initialization for a seed.
        if config.label_context_mode in {
            "previous_two_dwell_bucket",
            "previous_two_mode_dwell",
        }:
            self.history_sequence_dwell_embedding = nn.Embedding(
                7, config.label_context_feature_dim
            )
        if config.label_context_mode == "previous_two_mode_dwell":
            self.history_sequence_mode_embedding = nn.Embedding(
                3, config.label_context_feature_dim
            )
        if visual_residual:
            self.image_encoder = _arm_hold_image_encoder(config)
            self.visual_residual_head = nn.Sequential(
                nn.Linear(
                    feature_dim + config.image_feature_dim,
                    config.fusion_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Linear(config.fusion_dim, config.horizon * 2),
            )
            residual_output = self.visual_residual_head[-1]
            if not isinstance(residual_output, nn.Linear):
                raise AssertionError("Visual residual output must be linear.")
            nn.init.zeros_(residual_output.weight)
            nn.init.zeros_(residual_output.bias)

    def forward(
        self,
        *,
        front_rgb: torch.Tensor | None,
        observation_state: torch.Tensor | None,
        history_previous_mode: torch.Tensor | None = None,
        history_available_mask: torch.Tensor | None = None,
        history_dwell_steps: torch.Tensor | None = None,
        history_sequence_previous_mode: torch.Tensor | None = None,
        history_sequence_available_mask: torch.Tensor | None = None,
        history_sequence_dwell_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        image_feature: torch.Tensor | None = None
        if self.image_encoder is not None:
            if front_rgb is None or front_rgb.ndim != 4 or front_rgb.shape[1] != 3:
                raise ArmHoldBaselineError(
                    "Configured image encoder requires front_rgb [B,3,H,W]."
                )
            image_feature = self.image_encoder(front_rgb)
            if self.visual_residual_head is None:
                features.append(image_feature)
        elif front_rgb is not None:
            raise ArmHoldBaselineError("front_rgb was supplied to a state-only model.")
        if self.state_encoder is not None:
            if (
                observation_state is None
                or observation_state.ndim != 2
                or observation_state.shape[1] != self.config.state_dim
            ):
                raise ArmHoldBaselineError(
                    "Configured state encoder requires observation_state [B,state_dim]."
                )
            features.append(self.state_encoder(observation_state))
        elif observation_state is not None:
            raise ArmHoldBaselineError("state was supplied to a front-only model.")
        history_values = (
            history_previous_mode,
            history_available_mask,
            history_dwell_steps,
        )
        sequence_values = (
            history_sequence_previous_mode,
            history_sequence_available_mask,
            history_sequence_dwell_steps,
        )
        has_history = history_values[0] is not None
        has_sequence = sequence_values[0] is not None
        if any((value is not None) != has_history for value in history_values):
            raise ArmHoldBaselineError("History context inputs must be all-or-none.")
        if any((value is not None) != has_sequence for value in sequence_values):
            raise ArmHoldBaselineError(
                "History sequence inputs must be all-or-none."
            )
        if has_history and has_sequence:
            raise ArmHoldBaselineError(
                "One-step and two-step history inputs are mutually exclusive."
            )
        expects_sequence = self.config.label_context_mode.startswith(
            "previous_two_"
        )
        if self.config.label_context_mode == "none":
            if has_history or has_sequence:
                raise ArmHoldBaselineError(
                    "History context was supplied to a context-free model."
                )
        elif not expects_sequence:
            if has_sequence:
                raise ArmHoldBaselineError(
                    "One-step context model received two-step history."
                )
            if not has_history:
                raise ArmHoldBaselineError(
                    "Configured label context requires history tensors."
                )
            assert history_previous_mode is not None
            assert history_available_mask is not None
            assert history_dwell_steps is not None
            batch_size = history_previous_mode.shape[0]
            if (
                history_previous_mode.shape != (batch_size, 2)
                or history_available_mask.shape != (batch_size, 2)
                or history_dwell_steps.shape != (batch_size, 2)
            ):
                raise ArmHoldBaselineError("History context input shape is invalid.")
            context: torch.Tensor | None = None
            if self.history_mode_embedding is not None:
                mode_index = torch.where(
                    history_available_mask,
                    history_previous_mode.to(dtype=torch.int64) + 1,
                    torch.zeros_like(history_dwell_steps),
                )
                context = self.history_mode_embedding(mode_index)
            if self.history_dwell_embedding is not None:
                dwell_bucket = arm_hold_dwell_bucket_indices(
                    history_dwell_steps, history_available_mask
                )
                dwell_context = self.history_dwell_embedding(dwell_bucket)
                context = (
                    dwell_context if context is None else context + dwell_context
                )
            if context is None:
                raise AssertionError("Context model has no configured encoder.")
            features.append(context.reshape(batch_size, -1))
        else:
            if has_history:
                raise ArmHoldBaselineError(
                    "Two-step context model received one-step history."
                )
            if not has_sequence:
                raise ArmHoldBaselineError(
                    "Configured two-step context requires history tensors."
                )
            assert history_sequence_previous_mode is not None
            assert history_sequence_available_mask is not None
            assert history_sequence_dwell_steps is not None
            batch_size = history_sequence_previous_mode.shape[0]
            expected_shape = (batch_size, 2, 2)
            if (
                history_sequence_previous_mode.shape != expected_shape
                or history_sequence_available_mask.shape != expected_shape
                or history_sequence_dwell_steps.shape != expected_shape
            ):
                raise ArmHoldBaselineError(
                    "History sequence input shape is invalid."
                )
            if self.history_sequence_dwell_embedding is None:
                raise AssertionError("Two-step context has no dwell encoder.")
            dwell_bucket = arm_hold_dwell_bucket_indices(
                history_sequence_dwell_steps,
                history_sequence_available_mask,
            )
            context = self.history_sequence_dwell_embedding(dwell_bucket)
            if self.history_sequence_mode_embedding is not None:
                mode_index = torch.where(
                    history_sequence_available_mask,
                    history_sequence_previous_mode.to(dtype=torch.int64) + 1,
                    torch.zeros_like(history_sequence_dwell_steps),
                )
                context = context + self.history_sequence_mode_embedding(mode_index)
            features.append(context.reshape(batch_size, -1))
        if not features:
            raise ArmHoldBaselineError("Model has no configured input feature.")
        fused = features[0] if len(features) == 1 else torch.cat(features, dim=-1)
        logits = self.head(fused).reshape(
            fused.shape[0], self.config.horizon, 2
        )
        if self.visual_residual_head is None:
            return logits
        if image_feature is None:
            raise AssertionError("Visual residual model has no image feature.")
        residual = self.visual_residual_head(
            torch.cat((fused, image_feature), dim=-1)
        ).reshape(fused.shape[0], self.config.horizon, 2)
        return logits + residual

    def forward_batch(self, batch: ArmHoldTensorBatch) -> torch.Tensor:
        return self(
            front_rgb=batch.front_rgb,
            observation_state=batch.observation_state,
            history_previous_mode=batch.history_previous_mode,
            history_available_mask=batch.history_available_mask,
            history_dwell_steps=batch.history_dwell_steps,
            history_sequence_previous_mode=batch.history_sequence_previous_mode,
            history_sequence_available_mask=batch.history_sequence_available_mask,
            history_sequence_dwell_steps=batch.history_sequence_dwell_steps,
        )


def arm_hold_loss(
    logits: torch.Tensor,
    batch: ArmHoldTensorBatch,
    config: ArmHoldLossConfig = ArmHoldLossConfig(),
) -> torch.Tensor:
    """Masked BCE plus an optional stable-target probability persistence term."""

    if logits.shape != batch.target.shape:
        raise ArmHoldBaselineError("logits shape must match target shape.")
    mask = batch.valid_mask & batch.loss_mask
    if not bool(torch.any(mask)):
        raise ArmHoldBaselineError("Loss batch contains no valid target elements.")
    pos_weight = torch.tensor(
        [config.positive_weight_left, config.positive_weight_right],
        dtype=logits.dtype,
        device=logits.device,
    )
    element_loss = F.binary_cross_entropy_with_logits(
        logits,
        batch.target.to(device=logits.device, dtype=logits.dtype),
        reduction="none",
        pos_weight=pos_weight,
    )
    weights = torch.ones_like(element_loss)
    if config.exact_transition_weight != 1.0:
        weights = torch.where(
            batch.transition_mask.to(device=logits.device),
            torch.as_tensor(
                config.exact_transition_weight,
                dtype=logits.dtype,
                device=logits.device,
            ),
            weights,
        )
    effective = mask.to(device=logits.device, dtype=logits.dtype) * weights
    state_loss = torch.sum(element_loss * effective) / torch.sum(effective)
    if config.stable_persistence_weight == 0.0 or logits.shape[1] < 2:
        return state_loss
    target = batch.target.to(device=logits.device, dtype=logits.dtype)
    resolved_mask = mask.to(device=logits.device)
    covered_boundary = resolved_mask[:, 1:] & resolved_mask[:, :-1]
    derived_transition = target[:, 1:] != target[:, :-1]
    canonical_transition = batch.transition_mask[:, 1:].to(device=logits.device)
    if bool(
        torch.any(
            covered_boundary & (canonical_transition != derived_transition)
        )
    ):
        raise ArmHoldBaselineError(
            "transition_mask conflicts with adjacent effective targets."
        )
    stable_boundary = covered_boundary & ~canonical_transition
    if not bool(torch.any(stable_boundary)):
        return state_loss
    probabilities = torch.sigmoid(logits)
    persistence_error = torch.abs(probabilities[:, 1:] - probabilities[:, :-1])
    persistence_mask = stable_boundary.to(dtype=logits.dtype)
    persistence_loss = torch.sum(persistence_error * persistence_mask) / torch.sum(
        persistence_mask
    )
    return state_loss + config.stable_persistence_weight * persistence_loss


def arm_hold_fit_constant_side_logits(batch: ArmHoldTensorBatch) -> torch.Tensor:
    """Fit two constant logits; callers must fit on training data only."""

    mask = batch.valid_mask & batch.loss_mask
    counts = mask.sum(dim=(0, 1))
    if bool(torch.any(counts == 0)):
        raise ArmHoldBaselineError("Constant baseline has an uncovered side.")
    positives = (batch.target * mask).sum(dim=(0, 1))
    probability = (positives / counts).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def arm_hold_expand_constant_logits(
    side_logits: torch.Tensor,
    batch: ArmHoldTensorBatch,
) -> torch.Tensor:
    """Apply training-fitted constant logits to any compatible batch."""

    if side_logits.shape != (2,) or not bool(torch.all(torch.isfinite(side_logits))):
        raise ArmHoldBaselineError("side_logits must be finite with shape [2].")
    return side_logits.to(
        device=batch.target.device,
        dtype=batch.target.dtype,
    ).reshape(1, 1, 2).expand_as(batch.target).clone()


def arm_hold_constant_logits(batch: ArmHoldTensorBatch) -> torch.Tensor:
    """Fit and expand an input-free baseline on one batch.

    This convenience helper is intended for local checks.  Formal validation
    must fit with :func:`arm_hold_fit_constant_side_logits` on train data and
    apply with :func:`arm_hold_expand_constant_logits`.
    """

    return arm_hold_expand_constant_logits(
        arm_hold_fit_constant_side_logits(batch),
        batch,
    )


__all__ = [
    "ARM_HOLD_DWELL_BUCKET_THRESHOLDS",
    "ARM_HOLD_FUSION_MODES",
    "ARM_HOLD_LABEL_CONTEXT_MODES",
    "ArmHoldBaseline",
    "ArmHoldBaselineConfig",
    "ArmHoldBaselineError",
    "ArmHoldLossConfig",
    "ArmHoldTensorBatch",
    "arm_hold_constant_logits",
    "arm_hold_dwell_bucket_indices",
    "arm_hold_expand_constant_logits",
    "arm_hold_fit_constant_side_logits",
    "arm_hold_tensor_batch",
    "arm_hold_loss",
]
