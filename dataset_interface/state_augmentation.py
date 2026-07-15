"""Training-time augmentation for normalized continuous policy state.

This module intentionally has no torch dependency.  A model-specific training
adapter normalizes the base state, calls this module, then converts the result
to its native tensor type.  Deployment never samples noise or dropout; a
masked-state artifact receives the same base state with an all-one mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from dataset_contracts.state_input import (
    STATE_INPUT_MODE_RAW,
    StateInputMode,
    concatenate_state_and_mask,
    normalize_state_input_mode,
    uses_state_mask,
)
from dataset_contracts.v22 import PolicyIOLayout, SIDES


StateDropoutGranularity = Literal["whole_state", "side"]


@dataclass(frozen=True)
class StateAugmentationConfig:
    """Opt-in normalized-state noise and availability-mask dropout.

    When enabled, each continuous normalized-state element receives independent
    ``Normal(0, noise_std**2)`` noise.  Dropout is sampled independently per
    batch row (``whole_state``) or per batch-row/side (``side``), and happens
    after noise: ``state_aug = (state_norm + noise) * mask``.

    Discrete state tokens, actions, and arm-qpos audit columns are deliberately
    outside this augmentation.  The default is disabled for checkpoint and
    model-input backward compatibility.
    """

    enabled: bool = False
    noise_std: float = 0.02
    dropout_probability: float = 0.10
    dropout_granularity: StateDropoutGranularity = "whole_state"

    def __post_init__(self) -> None:
        if float(self.noise_std) < 0.0:
            raise ValueError("StateAugmentationConfig.noise_std must be non-negative.")
        if not 0.0 <= float(self.dropout_probability) <= 1.0:
            raise ValueError("StateAugmentationConfig.dropout_probability must be within [0, 1].")
        granularity = str(self.dropout_granularity).strip().lower()
        if granularity not in {"whole_state", "side"}:
            raise ValueError("StateAugmentationConfig.dropout_granularity must be 'whole_state' or 'side'.")
        object.__setattr__(self, "noise_std", float(self.noise_std))
        object.__setattr__(self, "dropout_probability", float(self.dropout_probability))
        object.__setattr__(self, "dropout_granularity", granularity)

    @property
    def requires_masked_state_input(self) -> bool:
        return bool(self.enabled and self.dropout_probability > 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "normalized_noise": {
                "distribution": "normal",
                "mean": 0.0,
                "std": float(self.noise_std),
                "per_element": True,
            },
            "dropout": {
                "probability": float(self.dropout_probability),
                "granularity": self.dropout_granularity,
                "formula": "state_aug=(state_norm+noise)*mask",
            },
            "excluded": ["action", "observation.state.discrete", "observation.arm.qpos"],
        }


@dataclass(frozen=True)
class StateAugmentationResult:
    """One augmented normalized-state batch and its binary availability mask."""

    state_augmented: np.ndarray
    mask: np.ndarray
    noise: np.ndarray

    def __post_init__(self) -> None:
        state = np.asarray(self.state_augmented, dtype=np.float32)
        mask = np.asarray(self.mask, dtype=np.float32)
        noise = np.asarray(self.noise, dtype=np.float32)
        if state.ndim not in (1, 2) or mask.shape != state.shape or noise.shape != state.shape:
            raise ValueError("state_augmented, mask, and noise must share [dim] or [batch, dim] shape.")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(noise)):
            raise ValueError("state augmentation values must be finite.")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("state augmentation mask must contain only 0 or 1.")
        object.__setattr__(self, "state_augmented", state.copy())
        object.__setattr__(self, "mask", mask.copy())
        object.__setattr__(self, "noise", noise.copy())

    def model_input(self, mode: StateInputMode | str = STATE_INPUT_MODE_RAW) -> np.ndarray:
        """Return D-dimensional or masked 2D-dimensional float32 model input."""

        normalized_mode = normalize_state_input_mode(mode)
        if not uses_state_mask(normalized_mode):
            return self.state_augmented.copy()
        return concatenate_state_and_mask(self.state_augmented, self.mask)


@dataclass(frozen=True)
class TorchStateAugmentationResult:
    """Torch counterpart kept lazy so the core interface has no torch import."""

    state_augmented: Any
    mask: Any
    noise: Any

    def model_input(self, mode: StateInputMode | str = STATE_INPUT_MODE_RAW) -> Any:
        torch = _require_torch()
        normalized_mode = normalize_state_input_mode(mode)
        if not uses_state_mask(normalized_mode):
            return self.state_augmented.clone()
        return torch.cat((self.state_augmented, self.mask), dim=-1)


def _as_batch(state_norm: Any, *, state_dim: int) -> tuple[np.ndarray, bool]:
    state = np.asarray(state_norm, dtype=np.float32)
    squeeze = state.ndim == 1
    if squeeze:
        state = state.reshape(1, -1)
    if state.ndim != 2 or state.shape[1] != int(state_dim):
        raise ValueError(
            f"normalized state must have shape ({int(state_dim)},) or [batch, {int(state_dim)}], "
            f"got {np.asarray(state_norm).shape}."
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("normalized state must be finite before augmentation.")
    return state, squeeze


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch normalized-state augmentation was requested, but torch is not installed."
        ) from exc
    return torch


def _as_torch_batch(state_norm: Any, *, state_dim: int) -> tuple[Any, bool]:
    torch = _require_torch()
    if not isinstance(state_norm, torch.Tensor):
        raise TypeError("state_norm must be a torch.Tensor for torch state augmentation.")
    if not state_norm.is_floating_point():
        raise TypeError("normalized torch state must have a floating dtype.")
    squeeze = state_norm.ndim == 1
    state = state_norm.unsqueeze(0) if squeeze else state_norm
    if state.ndim != 2 or state.shape[1] != int(state_dim):
        raise ValueError(
            f"normalized torch state must have shape ({int(state_dim)},) or [batch, {int(state_dim)}], "
            f"got {tuple(state_norm.shape)}."
        )
    if not bool(torch.isfinite(state).all()):
        raise ValueError("normalized torch state must be finite before augmentation.")
    return state, squeeze


def _side_component_slices(layout: PolicyIOLayout) -> dict[str, tuple[slice, ...]]:
    result: dict[str, list[slice]] = {side: [] for side in SIDES}
    for component in layout.components("state", "continuous"):
        if component.side not in result:
            raise ValueError(f"Unexpected continuous state component side {component.side!r}.")
        result[component.side].append(slice(component.offset, component.stop))
    return {side: tuple(slices) for side, slices in result.items()}


def augment_normalized_state(
    state_norm: Any,
    *,
    layout: PolicyIOLayout,
    config: StateAugmentationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> StateAugmentationResult:
    """Vectorize normalized-state augmentation over one training batch.

    ``rng`` should normally be the training worker's seeded NumPy generator.
    The operation is entirely in-memory and creates no dataset artifacts.
    """

    resolved = StateAugmentationConfig() if config is None else config
    state, squeeze = _as_batch(state_norm, state_dim=layout.state_dim)
    mask = np.ones_like(state, dtype=np.float32)
    noise = np.zeros_like(state, dtype=np.float32)
    if resolved.enabled:
        random = np.random.default_rng() if rng is None else rng
        if resolved.noise_std > 0.0:
            noise = random.normal(0.0, resolved.noise_std, size=state.shape).astype(np.float32)
        if resolved.dropout_probability > 0.0:
            if resolved.dropout_granularity == "whole_state":
                dropped = random.random(state.shape[0]) < resolved.dropout_probability
                mask[dropped, :] = 0.0
            else:
                component_slices = _side_component_slices(layout)
                dropped_by_side = random.random((state.shape[0], len(SIDES))) < resolved.dropout_probability
                for side_index, side in enumerate(SIDES):
                    for component_slice in component_slices[side]:
                        mask[dropped_by_side[:, side_index], component_slice] = 0.0
    augmented = (state + noise) * mask
    if squeeze:
        return StateAugmentationResult(
            state_augmented=augmented[0],
            mask=mask[0],
            noise=noise[0],
        )
    return StateAugmentationResult(state_augmented=augmented, mask=mask, noise=noise)


def augment_normalized_state_torch(
    state_norm: Any,
    *,
    layout: PolicyIOLayout,
    config: StateAugmentationConfig | None = None,
    generator: Any | None = None,
) -> TorchStateAugmentationResult:
    """GPU/CPU Torch equivalent of :func:`augment_normalized_state`.

    It is fully batch-vectorized and preserves the input tensor's device and
    floating dtype. ``generator`` should be the seeded generator owned by the
    training worker/process when reproducibility is required.
    """

    torch = _require_torch()
    resolved = StateAugmentationConfig() if config is None else config
    state, squeeze = _as_torch_batch(state_norm, state_dim=layout.state_dim)
    mask = torch.ones_like(state)
    noise = torch.zeros_like(state)
    if resolved.enabled:
        if resolved.noise_std > 0.0:
            noise = torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            ) * float(resolved.noise_std)
        if resolved.dropout_probability > 0.0:
            if resolved.dropout_granularity == "whole_state":
                dropped = torch.rand(
                    (state.shape[0],),
                    device=state.device,
                    generator=generator,
                ) < float(resolved.dropout_probability)
                mask[dropped, :] = 0.0
            else:
                component_slices = _side_component_slices(layout)
                dropped_by_side = torch.rand(
                    (state.shape[0], len(SIDES)),
                    device=state.device,
                    generator=generator,
                ) < float(resolved.dropout_probability)
                for side_index, side in enumerate(SIDES):
                    for component_slice in component_slices[side]:
                        mask[dropped_by_side[:, side_index], component_slice] = 0.0
    augmented = (state + noise) * mask
    if squeeze:
        return TorchStateAugmentationResult(
            state_augmented=augmented[0],
            mask=mask[0],
            noise=noise[0],
        )
    return TorchStateAugmentationResult(state_augmented=augmented, mask=mask, noise=noise)


class NormalizedStateInputAdapter:
    """Bridge a trainer's normalizer to the artifact-declared model input mode."""

    def __init__(
        self,
        *,
        layout: PolicyIOLayout,
        state_input_mode: StateInputMode | str = STATE_INPUT_MODE_RAW,
        config: StateAugmentationConfig | None = None,
    ) -> None:
        self.layout = layout
        self.state_input_mode = normalize_state_input_mode(state_input_mode)
        self.config = StateAugmentationConfig() if config is None else config
        if self.config.requires_masked_state_input and not uses_state_mask(self.state_input_mode):
            raise ValueError(
                "State dropout requires state_input_mode='masked_normalized_concat_v1' so the model "
                "can distinguish dropped state from a valid zero value."
            )

    @property
    def model_state_dim(self) -> int:
        return int(self.layout.state_dim) * (2 if uses_state_mask(self.state_input_mode) else 1)

    def apply(
        self,
        state_norm: Any,
        *,
        rng: np.random.Generator | None = None,
    ) -> StateAugmentationResult:
        return augment_normalized_state(
            state_norm,
            layout=self.layout,
            config=self.config,
            rng=rng,
        )

    def model_input(self, state_norm: Any, *, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.apply(state_norm, rng=rng).model_input(self.state_input_mode)

    def apply_torch(self, state_norm: Any, *, generator: Any | None = None) -> TorchStateAugmentationResult:
        return augment_normalized_state_torch(
            state_norm,
            layout=self.layout,
            config=self.config,
            generator=generator,
        )

    def torch_model_input(self, state_norm: Any, *, generator: Any | None = None) -> Any:
        return self.apply_torch(state_norm, generator=generator).model_input(self.state_input_mode)


__all__ = [
    "NormalizedStateInputAdapter",
    "StateAugmentationConfig",
    "StateAugmentationResult",
    "StateDropoutGranularity",
    "TorchStateAugmentationResult",
    "augment_normalized_state",
    "augment_normalized_state_torch",
]
