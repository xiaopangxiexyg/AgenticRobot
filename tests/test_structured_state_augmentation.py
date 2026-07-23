from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts.state_input import STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
from dataset_contracts.v22 import PolicyIOLayout, SIDES
from dataset_interface.state_augmentation import StateAugmentationConfig
from dataset_interface.structured_state_augmentation import (
    StructuredStateAugmentationConfig,
    StructuredStateInputAdapter,
    augment_structured_state,
)


_LAYOUT_PATH = (
    Path(__file__).parents[1]
    / "teleop_runtime/configs/policy_io_layouts/dual_arm_bimanual_dmtac_gripper_v22.json"
)


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout.from_dict(json.loads(_LAYOUT_PATH.read_text(encoding="utf-8")))


def _state(layout: PolicyIOLayout, *, batch: int | None = None) -> np.ndarray:
    value = np.zeros(layout.state_dim, dtype=np.float32)
    for component in layout.components("state", "continuous"):
        if component.name.endswith(".arm.pose"):
            value[component.offset : component.offset + 3] = (0.2, -0.1, 0.4)
            value[component.offset + 6] = 1.0
        else:
            value[component.offset : component.stop] = 0.5
    if batch is None:
        return value
    return np.repeat(value[None, :], batch, axis=0)


def test_disabled_structured_augmentation_is_an_exact_noop() -> None:
    layout = _layout()
    state = _state(layout)

    result = augment_structured_state(
        state,
        layout=layout,
        config=StructuredStateAugmentationConfig(enabled=False),
    )

    np.testing.assert_array_equal(result.state_augmented, state)
    np.testing.assert_array_equal(result.mask, np.ones_like(state))
    assert result.translation_noise_m.shape == (len(SIDES), 3)
    assert result.rotation_noise_rad.shape == (len(SIDES), 3)
    assert result.end_effector_noise.shape == (len(SIDES), 1)
    np.testing.assert_array_equal(result.translation_noise_m, 0.0)
    np.testing.assert_array_equal(result.rotation_noise_rad, 0.0)
    np.testing.assert_array_equal(result.end_effector_noise, 0.0)


def test_seeded_structured_augmentation_matches_historical_oracle() -> None:
    layout = _layout()

    result = augment_structured_state(
        _state(layout),
        layout=layout,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(
        result.translation_noise_m,
        np.asarray(
            [
                [-9.8912138e-04, -3.6778665e-04, 1.2879253e-03],
                [9.7167322e-05, -1.5259304e-03, 1.1921661e-03],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        result.rotation_noise_rad,
        np.asarray(
            [
                [0.00253912, 0.01204579, 0.00755427],
                [-0.00878454, 0.0130935, 0.00178444],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        result.end_effector_noise,
        np.asarray([[-0.00316595], [-0.00311795]], dtype=np.float32),
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_array_equal(result.mask, np.ones_like(result.mask))


def test_clip_scope_reuses_pose_noise_across_rows_but_keeps_effector_noise_per_row() -> None:
    layout = _layout()
    result = augment_structured_state(
        _state(layout, batch=3),
        layout=layout,
        config=StructuredStateAugmentationConfig(sample_scope="clip"),
        rng=np.random.default_rng(7),
    )

    for side_index in range(len(SIDES)):
        np.testing.assert_array_equal(
            result.translation_noise_m[:, side_index],
            np.repeat(result.translation_noise_m[:1, side_index], 3, axis=0),
        )
        np.testing.assert_array_equal(
            result.rotation_noise_rad[:, side_index],
            np.repeat(result.rotation_noise_rad[:1, side_index], 3, axis=0),
        )
    assert not np.array_equal(
        result.end_effector_noise[0], result.end_effector_noise[1]
    )


def test_structured_dropout_masks_only_declared_components_and_requires_mask_input() -> None:
    layout = _layout()
    config = StructuredStateAugmentationConfig(
        arm_translation_std_m=0.0,
        arm_rotation_std_deg=0.0,
        gripper_noise_std=0.0,
        arm_dropout_probability=1.0,
    )

    with pytest.raises(ValueError, match="Structured state dropout requires"):
        StructuredStateInputAdapter(layout=layout, structured_config=config)

    adapter = StructuredStateInputAdapter(
        layout=layout,
        structured_config=config,
        normalized_config=StateAugmentationConfig(enabled=False),
        state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    )
    raw = _state(layout)
    result = adapter.apply_raw(raw, rng=np.random.default_rng(4))

    for component in layout.components("state", "continuous"):
        component_mask = result.mask[component.offset : component.stop]
        expected = 0.0 if component.name.endswith(".arm.pose") else 1.0
        np.testing.assert_array_equal(
            component_mask,
            np.full(component.dim, expected, dtype=np.float32),
        )
    model_input = adapter.model_input(
        raw,
        lambda value: value,
        rng=np.random.default_rng(4),
    )
    assert model_input.shape == (layout.state_dim * 2,)
    np.testing.assert_array_equal(model_input[layout.state_dim :], result.mask)


def test_normalized_gripper_units_are_clipped_after_noise() -> None:
    layout = _layout()
    state = _state(layout)
    for component in layout.components("state", "continuous"):
        if not component.name.endswith(".arm.pose"):
            state[component.offset : component.stop] = 1.0

    result = augment_structured_state(
        state,
        layout=layout,
        config=StructuredStateAugmentationConfig(
            arm_translation_std_m=0.0,
            arm_rotation_std_deg=0.0,
            gripper_noise_std=100.0,
        ),
        rng=np.random.default_rng(31),
    )

    for component in layout.components("state", "continuous"):
        if not component.name.endswith(".arm.pose"):
            values = result.state_augmented[component.offset : component.stop]
            assert np.all(values >= 0.0)
            assert np.all(values <= 1.0)


def test_structured_config_serialization_preserves_versioned_contract() -> None:
    payload = StructuredStateAugmentationConfig().to_dict()

    assert payload["schema"] == "dexteleop_structured_state_augmentation_v1"
    assert payload["pose"] == {
        "space": "canonical_desk_se3",
        "translation_std_m": 0.001,
        "rotation_std_deg": 0.75,
        "composition": "delta_left_multiply",
    }
    assert payload["excluded"] == [
        "action",
        "observation.last_action",
        "observation.state.discrete",
        "observation.arm.qpos",
    ]
