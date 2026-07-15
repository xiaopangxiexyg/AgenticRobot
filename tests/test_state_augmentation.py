from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.state_input import STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
from dataset_contracts.v22 import PolicyIOLayout, SIDES
from dataset_interface import (
    NormalizedStateInputAdapter,
    StateAugmentationConfig,
    augment_normalized_state,
)


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout.legacy_v21_q6()


def test_disabled_state_augmentation_is_an_exact_noop_with_all_available_mask() -> None:
    layout = _layout()
    state = np.linspace(-1.0, 1.0, layout.state_dim, dtype=np.float32).reshape(1, -1)

    result = augment_normalized_state(state, layout=layout)

    np.testing.assert_array_equal(result.state_augmented, state)
    np.testing.assert_array_equal(result.mask, np.ones_like(state))
    np.testing.assert_array_equal(result.noise, np.zeros_like(state))
    np.testing.assert_array_equal(result.model_input(), state)


def test_normalized_noise_is_independent_per_element_and_seeded_by_training_rng() -> None:
    layout = _layout()
    state = np.zeros((2, layout.state_dim), dtype=np.float32)
    config = StateAugmentationConfig(enabled=True, noise_std=0.02, dropout_probability=0.0)
    expected_rng = np.random.default_rng(123)
    expected_noise = expected_rng.normal(0.0, 0.02, size=state.shape).astype(np.float32)

    result = augment_normalized_state(
        state,
        layout=layout,
        config=config,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(result.noise, expected_noise)
    np.testing.assert_array_equal(result.state_augmented, expected_noise)
    np.testing.assert_array_equal(result.mask, np.ones_like(state))


def test_whole_state_dropout_requires_masked_input_and_concatenates_binary_mask() -> None:
    layout = _layout()
    state = np.ones((3, layout.state_dim), dtype=np.float32)
    config = StateAugmentationConfig(enabled=True, noise_std=0.0, dropout_probability=1.0)

    with pytest.raises(ValueError, match="State dropout requires"):
        NormalizedStateInputAdapter(layout=layout, config=config)

    adapter = NormalizedStateInputAdapter(
        layout=layout,
        state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
        config=config,
    )
    result = adapter.apply(state, rng=np.random.default_rng(8))
    model_input = result.model_input(STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT)

    np.testing.assert_array_equal(result.state_augmented, np.zeros_like(state))
    np.testing.assert_array_equal(result.mask, np.zeros_like(state))
    assert model_input.shape == (3, layout.state_dim * 2)
    np.testing.assert_array_equal(model_input[:, : layout.state_dim], np.zeros_like(state))
    np.testing.assert_array_equal(model_input[:, layout.state_dim :], np.zeros_like(state))


def test_side_dropout_masks_every_continuous_component_for_that_side_only() -> None:
    layout = _layout()
    state = np.ones((2, layout.state_dim), dtype=np.float32)
    config = StateAugmentationConfig(
        enabled=True,
        noise_std=0.0,
        dropout_probability=0.5,
        dropout_granularity="side",
    )
    expected_rng = np.random.default_rng(71)
    dropped_by_side = expected_rng.random((state.shape[0], len(SIDES))) < config.dropout_probability

    result = augment_normalized_state(
        state,
        layout=layout,
        config=config,
        rng=np.random.default_rng(71),
    )

    for side_index, side in enumerate(SIDES):
        components = [
            component
            for component in layout.components("state", "continuous")
            if component.side == side
        ]
        assert components
        for row_index in range(state.shape[0]):
            expected_value = 0.0 if dropped_by_side[row_index, side_index] else 1.0
            for component in components:
                np.testing.assert_array_equal(
                    result.mask[row_index, component.offset : component.stop],
                    np.full((component.dim,), expected_value, dtype=np.float32),
                )


def test_torch_adapter_keeps_normalized_state_augmentation_on_the_tensor_device() -> None:
    torch = pytest.importorskip("torch")
    layout = _layout()
    state = torch.zeros((2, layout.state_dim), dtype=torch.float32)
    config = StateAugmentationConfig(enabled=True, noise_std=0.02, dropout_probability=0.0)
    adapter = NormalizedStateInputAdapter(
        layout=layout,
        state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
        config=config,
    )
    expected_generator = torch.Generator().manual_seed(123)
    expected_noise = torch.randn(state.shape, generator=expected_generator) * 0.02

    result = adapter.apply_torch(state, generator=torch.Generator().manual_seed(123))
    model_input = result.model_input(STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT)

    torch.testing.assert_close(result.noise, expected_noise)
    torch.testing.assert_close(result.state_augmented, expected_noise)
    torch.testing.assert_close(result.mask, torch.ones_like(state))
    assert model_input.shape == (2, layout.state_dim * 2)
    torch.testing.assert_close(model_input[:, layout.state_dim :], torch.ones_like(state))
