from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.arm_hold_history import (
    ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
)
from robot_motion.deploy.arm_hold import (
    ARM_HOLD_STATE_HOLD,
    ARM_HOLD_STATE_MOVING,
    ArmHoldResolutionReceipt,
    ArmHoldSideResolution,
)
from robot_motion.deploy.arm_hold_history import (
    ARM_HOLD_HISTORY_ADVANCED,
    ARM_HOLD_HISTORY_BINDING_MISMATCH,
    ARM_HOLD_HISTORY_DUPLICATE,
    ARM_HOLD_HISTORY_SEQUENCE_CONFLICT,
    ARM_HOLD_HISTORY_SEQUENCE_GAP,
    ARM_HOLD_HISTORY_STALE_SEQUENCE,
    ArmHoldResolvedHistoryBinding,
    ArmHoldResolvedHistoryError,
    ArmHoldResolvedHistoryState,
    ArmHoldResolvedHistoryUpdate,
    build_arm_hold_resolved_history_update,
)


def _binding(
    *, epoch: int = 0, binding_id: str = "policy-source"
) -> ArmHoldResolvedHistoryBinding:
    return ArmHoldResolvedHistoryBinding(
        binding_id=binding_id,
        epoch=epoch,
        contract_hash="hold-contract",
    )


def _resolution(
    combination: str,
    *,
    contract_hash: str = "hold-contract",
) -> ArmHoldResolutionReceipt:
    states = tuple(
        ARM_HOLD_STATE_HOLD if value == "1" else ARM_HOLD_STATE_MOVING
        for value in combination
    )
    return ArmHoldResolutionReceipt(
        contract_hash=contract_hash,
        source_combination=combination,
        resolved_combination=combination,
        side_resolutions=tuple(
            ArmHoldSideResolution(
                side=side,
                probability=float(value),
                previous_state=ARM_HOLD_STATE_MOVING,
                resolved_state=state,
                transition=(
                    "enter_hold" if state == ARM_HOLD_STATE_HOLD else "stay_moving"
                ),
                consecutive_count=1,
                reference_operation=(
                    "latched" if state == ARM_HOLD_STATE_HOLD else "none"
                ),
            )
            for side, value, state in zip(
                ("left", "right"), combination, states
            )
        ),
        reference_observation_step_index=1,
    )


def _update(
    binding: ArmHoldResolvedHistoryBinding,
    sequence: int,
    combination: str,
    *,
    suffix: str = "",
) -> ArmHoldResolvedHistoryUpdate:
    return build_arm_hold_resolved_history_update(
        binding=binding,
        resolution_sequence=sequence,
        resolution_id=f"resolution:{sequence}{suffix}",
        resolution=_resolution(combination),
    )


def test_binding_activation_bootstraps_explicit_unavailable_context() -> None:
    state = ArmHoldResolvedHistoryState()

    with pytest.raises(RuntimeError, match="no active binding"):
        state.snapshot()
    snapshot = state.activate(_binding())

    assert snapshot.context.provenance == ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1
    assert snapshot.context.available_mask.tolist() == [False, False]
    assert snapshot.context.previous_mode.tolist() == [0.0, 0.0]
    assert snapshot.context.dwell_steps.tolist() == [0, 0]
    assert snapshot.last_resolution_sequence is None


def test_free_running_history_exposes_only_the_previous_committed_resolution() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    sequence = (
        (10, "00", [0.0, 0.0], [1, 1]),
        (11, "10", [1.0, 0.0], [1, 2]),
        (12, "10", [1.0, 0.0], [2, 3]),
        (13, "11", [1.0, 1.0], [3, 1]),
    )

    expected_before = None
    for resolution_sequence, combination, modes, dwell in sequence:
        before = state.snapshot().context
        if expected_before is None:
            assert before.available_mask.tolist() == [False, False]
        else:
            assert before.previous_mode.tolist() == expected_before
        receipt = state.advance(_update(binding, resolution_sequence, combination))
        after = state.snapshot().context
        assert receipt.status == ARM_HOLD_HISTORY_ADVANCED
        assert receipt.accepted is True
        assert receipt.mutated is True
        assert after.previous_mode.tolist() == modes
        assert after.dwell_steps.tolist() == dwell
        expected_before = modes


def test_duplicate_is_idempotent_and_conflict_does_not_mutate() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    update = _update(binding, 4, "10")
    state.advance(update)

    duplicate = state.advance(update)
    conflict = state.advance(_update(binding, 4, "01", suffix=":conflict"))
    snapshot = state.snapshot()

    assert duplicate.status == ARM_HOLD_HISTORY_DUPLICATE
    assert duplicate.accepted is True
    assert duplicate.mutated is False
    assert conflict.status == ARM_HOLD_HISTORY_SEQUENCE_CONFLICT
    assert conflict.accepted is False
    assert snapshot.context.previous_mode.tolist() == [1.0, 0.0]
    assert snapshot.context.dwell_steps.tolist() == [1, 1]


def test_gap_and_stale_updates_are_rejected_but_can_be_recovered_in_order() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    state.advance(_update(binding, 10, "00"))
    future = _update(binding, 12, "11")

    gap = state.advance(future)
    middle = state.advance(_update(binding, 11, "10"))
    recovered = state.advance(future)
    stale = state.advance(_update(binding, 9, "01"))

    assert gap.status == ARM_HOLD_HISTORY_SEQUENCE_GAP
    assert gap.accepted is False
    assert middle.status == ARM_HOLD_HISTORY_ADVANCED
    assert recovered.status == ARM_HOLD_HISTORY_ADVANCED
    assert stale.status == ARM_HOLD_HISTORY_STALE_SEQUENCE
    assert stale.accepted is False
    assert state.snapshot().context.previous_mode.tolist() == [1.0, 1.0]


def test_epoch_rotation_resets_and_rejects_updates_from_the_old_binding() -> None:
    old_binding = _binding(epoch=2)
    new_binding = _binding(epoch=3)
    state = ArmHoldResolvedHistoryState()
    state.activate(old_binding)
    old_update = _update(old_binding, 8, "10")
    state.advance(old_update)

    reset = state.activate(new_binding)
    stale_binding = state.advance(old_update)
    first_new = state.advance(_update(new_binding, 20, "01"))
    idempotent_activation = state.activate(new_binding)

    assert reset.context.available_mask.tolist() == [False, False]
    assert stale_binding.status == ARM_HOLD_HISTORY_BINDING_MISMATCH
    assert stale_binding.accepted is False
    assert first_new.status == ARM_HOLD_HISTORY_ADVANCED
    assert idempotent_activation.context.previous_mode.tolist() == [0.0, 1.0]
    with pytest.raises(ArmHoldResolvedHistoryError, match="older epoch"):
        state.activate(old_binding)


def test_update_fails_closed_on_contract_or_resolution_contradiction() -> None:
    binding = _binding()
    with pytest.raises(ArmHoldResolvedHistoryError, match="contract hash"):
        build_arm_hold_resolved_history_update(
            binding=binding,
            resolution_sequence=1,
            resolution_id="resolution:1",
            resolution=_resolution("00", contract_hash="other-contract"),
        )

    resolution = _resolution("10")
    contradictory = ArmHoldResolutionReceipt(
        contract_hash=resolution.contract_hash,
        source_combination=resolution.source_combination,
        resolved_combination="01",
        side_resolutions=resolution.side_resolutions,
        reference_observation_step_index=1,
    )
    with pytest.raises(ArmHoldResolvedHistoryError, match="disagrees"):
        ArmHoldResolvedHistoryUpdate(
            binding_hash=binding.semantic_content_hash,
            resolution_sequence=1,
            resolution_id="resolution:1",
            resolution=contradictory,
        )


def test_state_rechecks_resolution_contract_without_mutating_history() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    state.advance(_update(binding, 1, "10"))
    contradictory = ArmHoldResolvedHistoryUpdate(
        binding_hash=binding.semantic_content_hash,
        resolution_sequence=2,
        resolution_id="resolution:2",
        resolution=_resolution("01", contract_hash="other-contract"),
    )

    receipt = state.advance(contradictory)
    snapshot = state.snapshot()

    assert receipt.status == ARM_HOLD_HISTORY_BINDING_MISMATCH
    assert receipt.accepted is False
    assert receipt.mutated is False
    assert snapshot.last_resolution_sequence == 1
    assert snapshot.context.previous_mode.tolist() == [1.0, 0.0]
    assert snapshot.context.dwell_steps.tolist() == [1, 1]


def test_deactivate_discards_binding_and_resolved_history() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    state.advance(_update(binding, 1, "11"))

    state.deactivate()

    with pytest.raises(RuntimeError, match="no active binding"):
        state.snapshot()
    reset = state.activate(binding)
    assert reset.context.available_mask.tolist() == [False, False]
    assert reset.context.dwell_steps.tolist() == [0, 0]


def test_context_arrays_remain_immutable_after_state_advancement() -> None:
    binding = _binding()
    state = ArmHoldResolvedHistoryState()
    state.activate(binding)
    state.advance(_update(binding, 1, "10"))
    context = state.snapshot().context

    with pytest.raises(ValueError, match="read-only"):
        context.dwell_steps[0] = 99
    np.testing.assert_array_equal(
        state.snapshot().context.dwell_steps,
        np.asarray([1, 1]),
    )
