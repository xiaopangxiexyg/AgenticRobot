from __future__ import annotations

import numpy as np
import pytest
import torch

from dataset_interface.arm_hold_windows import (
    ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1,
    ArmHoldHistoryContext,
    ArmHoldWindowSample,
)
from dataset_contracts.arm_hold_history import (
    ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    ArmHoldHistorySequenceContext,
)
from experiments.policy_learning.arm_hold import (
    ArmHoldBaseline,
    ArmHoldBaselineConfig,
    ArmHoldLossConfig,
    ArmHoldTensorBatch,
    arm_hold_constant_logits,
    arm_hold_dwell_bucket_indices,
    arm_hold_expand_constant_logits,
    arm_hold_fit_constant_side_logits,
    arm_hold_tensor_batch,
    arm_hold_loss,
)


def _sample(
    value: int,
    *,
    front: bool = True,
    state: bool = True,
    history: bool = False,
    history_sequence: bool = False,
) -> ArmHoldWindowSample:
    if history and history_sequence:
        raise AssertionError("Test sample history modes are mutually exclusive.")
    target = np.asarray(
        [
            [value, 1 - value],
            [value, 1 - value],
            [1 - value, value],
        ],
        dtype=np.float32,
    )
    transition = np.zeros((3, 2), dtype=bool)
    transition[2] = True
    return ArmHoldWindowSample(
        front_rgb=(
            None
            if not front
            else np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
        ),
        observation_state=(
            None
            if not state
            else np.asarray([value, 1 - value, 0.25, -0.25], dtype=np.float32)
        ),
        arm_hold_target=target,
        arm_hold_valid_mask=np.ones((3, 2), dtype=bool),
        arm_hold_loss_mask=np.ones((3, 2), dtype=bool),
        arm_hold_transition_mask=transition,
        arm_hold_history_context=(
            ArmHoldHistoryContext(
                previous_mode=np.asarray([value, 1 - value], dtype=np.float32),
                available_mask=np.asarray([True, True]),
                dwell_steps=np.asarray([2, 5], dtype=np.int64),
            )
            if history
            else None
        ),
        arm_hold_history_sequence_context=(
            ArmHoldHistorySequenceContext(
                previous_mode=np.asarray(
                    [[value, 1 - value], [value, 1 - value]], dtype=np.float32
                ),
                available_mask=np.ones((2, 2), dtype=bool),
                dwell_steps=np.asarray([[1, 4], [2, 5]], dtype=np.int64),
            )
            if history_sequence
            else None
        ),
        metadata={"sample": value},
    )


def test_collate_normalizes_rgb_and_preserves_masks() -> None:
    batch = arm_hold_tensor_batch((_sample(0), _sample(1)))

    assert batch.front_rgb.shape == (2, 3, 8, 8)
    assert batch.observation_state.shape == (2, 4)
    assert batch.target.shape == (2, 3, 2)
    assert batch.front_rgb[0, :, 0, 0].tolist() == pytest.approx(
        [10 / 255.0, 20 / 255.0, 30 / 255.0]
    )
    assert batch.valid_mask.all()
    assert batch.loss_mask.all()
    assert batch.transition_mask[:, 2].all()


def test_collate_preserves_typed_history_context() -> None:
    batch = arm_hold_tensor_batch(
        (_sample(0, history=True), _sample(1, history=True))
    )

    assert batch.history_previous_mode.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert batch.history_available_mask.tolist() == [[True, True], [True, True]]
    assert batch.history_dwell_steps.tolist() == [[2, 5], [2, 5]]
    assert (
        batch.history_context_provenance
        == ARM_HOLD_HISTORY_CONTEXT_PROVENANCE_V1
    )


def test_collate_rejects_mixed_history_context_availability() -> None:
    with pytest.raises(ValueError, match="mixes"):
        arm_hold_tensor_batch((_sample(0, history=True), _sample(1)))


def test_collate_preserves_typed_history_sequence_context() -> None:
    batch = arm_hold_tensor_batch(
        (_sample(0, history_sequence=True), _sample(1, history_sequence=True))
    )

    assert batch.history_previous_mode is None
    assert batch.history_sequence_previous_mode.shape == (2, 2, 2)
    assert batch.history_sequence_previous_mode[0].tolist() == [
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert batch.history_sequence_dwell_steps[0].tolist() == [[1, 4], [2, 5]]


def test_dwell_bucket_contract_is_explicit_and_bounded() -> None:
    dwell = torch.tensor([[0, 1, 2, 3, 4, 7, 8, 15, 16, 99]])
    available = dwell > 0

    assert arm_hold_dwell_bucket_indices(dwell, available).tolist() == [
        [0, 1, 2, 3, 4, 4, 5, 5, 6, 6]
    ]


@pytest.mark.parametrize("mode", ["state_only", "front_only", "front_state"])
def test_baseline_modes_produce_parallel_horizon_logits(mode: str) -> None:
    use_front = mode in {"front_only", "front_state"}
    use_state = mode in {"state_only", "front_state"}
    batch = arm_hold_tensor_batch(
        (
            _sample(0, front=use_front, state=use_state),
            _sample(1, front=use_front, state=use_state),
        )
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(input_mode=mode, state_dim=4, horizon=3)
    )

    logits = model.forward_batch(batch)

    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()


def test_visual_residual_requires_front_state_input() -> None:
    with pytest.raises(ValueError, match="requires front_state"):
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            fusion_mode="state_history_visual_residual_v1",
        )


def test_visual_residual_preserves_seeded_state_history_baseline_at_init() -> None:
    common = dict(
        state_dim=4,
        horizon=3,
        state_feature_dim=8,
        fusion_dim=12,
        label_context_mode="previous_dwell_bucket",
        label_context_feature_dim=4,
    )
    torch.manual_seed(47)
    state_history = ArmHoldBaseline(
        ArmHoldBaselineConfig(input_mode="state_only", **common)
    )
    torch.manual_seed(47)
    visual_residual = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="front_state",
            fusion_mode="state_history_visual_residual_v1",
            **common,
        )
    )

    residual_state = visual_residual.state_dict()
    for name, value in state_history.state_dict().items():
        assert name in residual_state
        assert torch.equal(value, residual_state[name])

    state_batch = arm_hold_tensor_batch(
        (_sample(0, front=False, history=True), _sample(1, front=False, history=True))
    )
    visual_batch = arm_hold_tensor_batch(
        (_sample(0, history=True), _sample(1, history=True))
    )
    assert torch.equal(
        state_history.forward_batch(state_batch),
        visual_residual.forward_batch(visual_batch),
    )
    assert visual_residual.visual_residual_head is not None
    residual_output = visual_residual.visual_residual_head[-1]
    assert isinstance(residual_output, torch.nn.Linear)
    assert not torch.any(residual_output.weight)
    assert not torch.any(residual_output.bias)


def test_visual_residual_output_branch_receives_training_gradient() -> None:
    batch = arm_hold_tensor_batch(
        (_sample(0, history=True), _sample(1, history=True))
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="front_state",
            state_dim=4,
            horizon=3,
            state_feature_dim=8,
            fusion_dim=12,
            label_context_mode="previous_dwell_bucket",
            label_context_feature_dim=4,
            fusion_mode="state_history_visual_residual_v1",
        )
    )
    loss = arm_hold_loss(model.forward_batch(batch), batch)

    loss.backward()

    assert model.visual_residual_head is not None
    residual_output = model.visual_residual_head[-1]
    assert isinstance(residual_output, torch.nn.Linear)
    assert residual_output.weight.grad is not None
    assert torch.isfinite(residual_output.weight.grad).all()
    assert torch.any(residual_output.weight.grad != 0)


@pytest.mark.parametrize(
    "context_mode",
    ["previous_mode", "previous_dwell_bucket", "previous_mode_dwell"],
)
def test_context_modes_produce_parallel_horizon_logits(context_mode: str) -> None:
    batch = arm_hold_tensor_batch(
        (
            _sample(0, front=False, history=True),
            _sample(1, front=False, history=True),
        )
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            label_context_mode=context_mode,
        )
    )

    logits = model.forward_batch(batch)

    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "context_mode",
    ["previous_two_dwell_bucket", "previous_two_mode_dwell"],
)
def test_two_step_context_modes_produce_parallel_horizon_logits(
    context_mode: str,
) -> None:
    batch = arm_hold_tensor_batch(
        (
            _sample(0, front=False, history_sequence=True),
            _sample(1, front=False, history_sequence=True),
        )
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            label_context_mode=context_mode,
        )
    )

    logits = model.forward_batch(batch)

    assert logits.shape == (2, 3, 2)
    assert torch.isfinite(logits).all()


def test_context_requirement_fails_closed_in_both_directions() -> None:
    ordinary = arm_hold_tensor_batch((_sample(0, front=False),))
    contextual = arm_hold_tensor_batch(
        (_sample(0, front=False, history=True),)
    )
    without_context = ArmHoldBaseline(
        ArmHoldBaselineConfig(input_mode="state_only", state_dim=4, horizon=3)
    )
    with_context = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            label_context_mode="previous_mode",
        )
    )

    with pytest.raises(ValueError, match="context-free"):
        without_context.forward_batch(contextual)
    with pytest.raises(ValueError, match="requires history"):
        with_context.forward_batch(ordinary)


def test_context_model_accepts_typed_resolved_history_provenance() -> None:
    source = arm_hold_tensor_batch(
        (_sample(0, front=False, history=True),)
    )
    resolved = ArmHoldTensorBatch(
        front_rgb=source.front_rgb,
        observation_state=source.observation_state,
        target=source.target,
        valid_mask=source.valid_mask,
        loss_mask=source.loss_mask,
        transition_mask=source.transition_mask,
        metadata=source.metadata,
        history_previous_mode=source.history_previous_mode,
        history_available_mask=source.history_available_mask,
        history_dwell_steps=source.history_dwell_steps,
        history_context_provenance=ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            label_context_mode="previous_mode",
        )
    )

    assert model.forward_batch(resolved).shape == (1, 3, 2)


def test_mode_and_mode_dwell_share_seeded_common_initialization() -> None:
    common = dict(
        input_mode="state_only",
        state_dim=4,
        horizon=3,
        state_feature_dim=8,
        fusion_dim=12,
    )
    torch.manual_seed(19)
    mode_only = ArmHoldBaseline(
        ArmHoldBaselineConfig(**common, label_context_mode="previous_mode")
    )
    torch.manual_seed(19)
    mode_dwell = ArmHoldBaseline(
        ArmHoldBaselineConfig(**common, label_context_mode="previous_mode_dwell")
    )

    for left, right in zip(
        mode_only.state_dict().items(), mode_dwell.state_dict().items()
    ):
        if left[0].startswith("history_dwell_embedding") or right[0].startswith(
            "history_dwell_embedding"
        ):
            continue
        assert left[0] == right[0]
        assert torch.equal(left[1], right[1])


def test_two_step_variants_share_seeded_common_initialization() -> None:
    common = dict(
        input_mode="state_only",
        state_dim=4,
        horizon=3,
        state_feature_dim=8,
        fusion_dim=12,
    )
    torch.manual_seed(41)
    dwell_only = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            **common, label_context_mode="previous_two_dwell_bucket"
        )
    )
    torch.manual_seed(41)
    mode_dwell = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            **common, label_context_mode="previous_two_mode_dwell"
        )
    )

    dwell_state = dwell_only.state_dict()
    mode_state = mode_dwell.state_dict()
    assert set(mode_state) == set(dwell_state) | {
        "history_sequence_mode_embedding.weight"
    }
    for name, value in dwell_state.items():
        assert torch.equal(value, mode_state[name])


def test_dwell_only_affects_the_explicit_dwell_context_mode() -> None:
    common = dict(
        input_mode="state_only",
        state_dim=4,
        horizon=3,
        state_feature_dim=8,
        fusion_dim=12,
    )
    state = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    previous = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    available = torch.tensor([[True, True]])
    short_dwell = torch.tensor([[1, 1]], dtype=torch.int64)
    long_dwell = torch.tensor([[8, 16]], dtype=torch.int64)

    torch.manual_seed(29)
    mode_only = ArmHoldBaseline(
        ArmHoldBaselineConfig(**common, label_context_mode="previous_mode")
    )
    torch.manual_seed(29)
    mode_dwell = ArmHoldBaseline(
        ArmHoldBaselineConfig(**common, label_context_mode="previous_mode_dwell")
    )

    def predict(model: ArmHoldBaseline, dwell: torch.Tensor) -> torch.Tensor:
        return model(
            front_rgb=None,
            observation_state=state,
            history_previous_mode=previous,
            history_available_mask=available,
            history_dwell_steps=dwell,
        )

    assert torch.equal(predict(mode_only, short_dwell), predict(mode_only, long_dwell))
    assert not torch.equal(
        predict(mode_dwell, short_dwell), predict(mode_dwell, long_dwell)
    )


def test_dwell_bucket_context_encodes_dwell_but_not_previous_mode() -> None:
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            state_feature_dim=8,
            fusion_dim=12,
            label_context_mode="previous_dwell_bucket",
        )
    )
    state = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    available = torch.tensor([[True, True]])
    short_dwell = torch.tensor([[1, 2]], dtype=torch.int64)
    long_dwell = torch.tensor([[8, 16]], dtype=torch.int64)

    def predict(previous: torch.Tensor, dwell: torch.Tensor) -> torch.Tensor:
        return model(
            front_rgb=None,
            observation_state=state,
            history_previous_mode=previous,
            history_available_mask=available,
            history_dwell_steps=dwell,
        )

    short_from_teleop = predict(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32), short_dwell
    )
    short_from_hold = predict(
        torch.tensor([[1.0, 1.0]], dtype=torch.float32), short_dwell
    )
    assert torch.equal(short_from_teleop, short_from_hold)
    assert not torch.equal(
        short_from_teleop,
        predict(torch.tensor([[0.0, 0.0]], dtype=torch.float32), long_dwell),
    )
    assert model.history_mode_embedding is None
    assert model.history_dwell_embedding is not None


def test_two_step_dwell_only_ignores_modes_but_full_context_consumes_them() -> None:
    common = dict(
        input_mode="state_only",
        state_dim=4,
        horizon=3,
        state_feature_dim=8,
        fusion_dim=12,
    )
    torch.manual_seed(43)
    dwell_only = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            **common, label_context_mode="previous_two_dwell_bucket"
        )
    )
    torch.manual_seed(43)
    mode_dwell = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            **common, label_context_mode="previous_two_mode_dwell"
        )
    )
    state = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    available = torch.ones((1, 2, 2), dtype=torch.bool)
    dwell = torch.tensor([[[3, 8], [4, 9]]], dtype=torch.int64)
    zeros = torch.zeros((1, 2, 2), dtype=torch.float32)
    ones = torch.ones((1, 2, 2), dtype=torch.float32)

    def predict(model: ArmHoldBaseline, mode: torch.Tensor) -> torch.Tensor:
        return model(
            front_rgb=None,
            observation_state=state,
            history_sequence_previous_mode=mode,
            history_sequence_available_mask=available,
            history_sequence_dwell_steps=dwell,
        )

    assert torch.equal(predict(dwell_only, zeros), predict(dwell_only, ones))
    assert not torch.equal(predict(mode_dwell, zeros), predict(mode_dwell, ones))


def test_weighted_bce_is_finite_and_backpropagates() -> None:
    batch = arm_hold_tensor_batch((_sample(0), _sample(1)))
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(input_mode="front_state", state_dim=4, horizon=3)
    )
    logits = model.forward_batch(batch)
    loss = arm_hold_loss(
        logits,
        batch,
        ArmHoldLossConfig(
            positive_weight_left=1.5,
            positive_weight_right=0.75,
            exact_transition_weight=3.0,
        ),
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_zero_persistence_weight_preserves_existing_loss_exactly() -> None:
    batch = arm_hold_tensor_batch((_sample(0), _sample(1)))
    logits = torch.linspace(-2.0, 2.0, batch.target.numel()).reshape_as(
        batch.target
    )

    ordinary = arm_hold_loss(logits, batch)
    explicit_zero = arm_hold_loss(
        logits,
        batch,
        ArmHoldLossConfig(stable_persistence_weight=0.0),
    )

    assert torch.equal(ordinary, explicit_zero)


def test_persistence_penalizes_probability_change_only_on_stable_targets() -> None:
    batch = arm_hold_tensor_batch((_sample(0),))
    stable_jump = torch.tensor(
        [[[-4.0, 4.0], [4.0, -4.0], [4.0, -4.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    transition_only = torch.where(batch.target.bool(), 4.0, -4.0)
    config = ArmHoldLossConfig(stable_persistence_weight=0.1)

    regularized = arm_hold_loss(stable_jump, batch, config)
    assert regularized > arm_hold_loss(stable_jump, batch)
    regularized.backward()
    assert stable_jump.grad is not None
    assert torch.isfinite(stable_jump.grad).all()
    assert torch.equal(
        arm_hold_loss(transition_only, batch, config),
        arm_hold_loss(transition_only, batch),
    )


def test_persistence_does_not_bridge_masked_boundaries() -> None:
    source = arm_hold_tensor_batch((_sample(0),))
    loss_mask = source.loss_mask.clone()
    loss_mask[:, 1, 0] = False
    batch = ArmHoldTensorBatch(
        front_rgb=source.front_rgb,
        observation_state=source.observation_state,
        target=source.target,
        valid_mask=source.valid_mask,
        loss_mask=loss_mask,
        transition_mask=source.transition_mask,
        metadata=source.metadata,
    )
    logits = torch.where(source.target.bool(), 4.0, -4.0)
    logits[:, 0, 0] = -4.0
    logits[:, 1, 0] = 4.0
    config = ArmHoldLossConfig(stable_persistence_weight=0.1)

    assert torch.equal(
        arm_hold_loss(logits, batch, config),
        arm_hold_loss(logits, batch),
    )


def test_persistence_fails_closed_on_transition_topology_conflict() -> None:
    source = arm_hold_tensor_batch((_sample(0),))
    transition_mask = source.transition_mask.clone()
    transition_mask[:, 1] = True
    batch = ArmHoldTensorBatch(
        front_rgb=source.front_rgb,
        observation_state=source.observation_state,
        target=source.target,
        valid_mask=source.valid_mask,
        loss_mask=source.loss_mask,
        transition_mask=transition_mask,
        metadata=source.metadata,
    )

    with pytest.raises(ValueError, match="conflicts"):
        arm_hold_loss(
            torch.zeros_like(source.target),
            batch,
            ArmHoldLossConfig(stable_persistence_weight=0.1),
        )


def test_persistence_weight_must_be_nonnegative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ArmHoldLossConfig(stable_persistence_weight=-0.1)


def test_loss_mask_excludes_elements_and_rejects_empty_batch() -> None:
    source = arm_hold_tensor_batch((_sample(0, front=False),))
    partial_mask = source.loss_mask.clone()
    partial_mask[:, 0] = False
    partial = ArmHoldTensorBatch(
        front_rgb=None,
        observation_state=source.observation_state,
        target=source.target,
        valid_mask=source.valid_mask,
        loss_mask=partial_mask,
        transition_mask=source.transition_mask,
        metadata=source.metadata,
    )
    logits = torch.zeros_like(source.target, requires_grad=True)

    loss = arm_hold_loss(logits, partial)

    assert loss.item() == pytest.approx(float(np.log(2.0)))
    empty = ArmHoldTensorBatch(
        front_rgb=None,
        observation_state=source.observation_state,
        target=source.target,
        valid_mask=source.valid_mask,
        loss_mask=torch.zeros_like(source.loss_mask),
        transition_mask=source.transition_mask,
        metadata=source.metadata,
    )
    with pytest.raises(ValueError, match="no valid"):
        arm_hold_loss(logits, empty)


def test_constant_baseline_is_input_free_and_matches_side_prevalence() -> None:
    batch = arm_hold_tensor_batch((_sample(0), _sample(0), _sample(1)))

    logits = arm_hold_constant_logits(batch)
    probability = torch.sigmoid(logits[0, 0])

    expected = batch.target.mean(dim=(0, 1))
    assert logits.shape == batch.target.shape
    assert probability.tolist() == pytest.approx(expected.tolist())


def test_constant_baseline_can_fit_train_and_apply_to_validation() -> None:
    train = arm_hold_tensor_batch((_sample(0), _sample(0), _sample(1)))
    validation = arm_hold_tensor_batch((_sample(1),))

    fitted = arm_hold_fit_constant_side_logits(train)
    applied = arm_hold_expand_constant_logits(fitted, validation)

    assert applied.shape == validation.target.shape
    assert torch.sigmoid(applied[0, 0]).tolist() == pytest.approx(
        train.target.mean(dim=(0, 1)).tolist()
    )


def test_constant_baseline_rejects_invalid_fitted_logits() -> None:
    batch = arm_hold_tensor_batch((_sample(0),))

    with pytest.raises(ValueError, match="shape"):
        arm_hold_expand_constant_logits(torch.zeros(1), batch)


def test_state_only_single_batch_can_overfit() -> None:
    torch.manual_seed(7)
    batch = arm_hold_tensor_batch(
        (
            _sample(0, front=False),
            _sample(1, front=False),
        )
    )
    model = ArmHoldBaseline(
        ArmHoldBaselineConfig(
            input_mode="state_only",
            state_dim=4,
            horizon=3,
            state_feature_dim=16,
            fusion_dim=24,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = arm_hold_loss(
        model.forward_batch(batch), batch
    ).detach().item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = arm_hold_loss(model.forward_batch(batch), batch)
        loss.backward()
        optimizer.step()
    final = arm_hold_loss(
        model.forward_batch(batch), batch
    ).detach().item()

    assert final < initial * 0.1
