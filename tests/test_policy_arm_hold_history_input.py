from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.arm_hold_history import (
    ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1,
    ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    POLICY_ARM_HOLD_HISTORY_INPUT_HASH_KEY,
    POLICY_ARM_HOLD_HISTORY_INPUT_METADATA_KEY,
    ArmHoldHistoryContext,
    ArmHoldHistoryContextError,
    PolicyArmHoldHistoryInputContract,
    encode_policy_arm_hold_history_input,
)
from dataset_contracts.arm_mode import DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1
from dataset_contracts.policy_action_semantics import PolicyArmHoldOutputContract
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.v22 import (
    DatasetLayoutBindingError,
    EndEffectorSpec,
    PolicyIOLayout,
)


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="arm_hold_history_input_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )


def _output_contract() -> PolicyArmHoldOutputContract:
    return PolicyArmHoldOutputContract(
        output_key="arm_hold_probability",
        training_contract_hash=(
            DEFAULT_ARM_HOLD_TRAINING_CONTRACT_V1.semantic_content_hash
        ),
        enter_threshold=0.8,
        exit_threshold=0.2,
        enter_consecutive_steps=1,
        exit_consecutive_steps=1,
    )


def _input_contract() -> PolicyArmHoldHistoryInputContract:
    return PolicyArmHoldHistoryInputContract(
        input_key="arm_hold_history_dwell_bucket"
    )


@pytest.mark.parametrize(
    ("dwell", "expected"),
    ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (7, 4), (8, 5), (15, 5), (16, 6)),
)
@pytest.mark.parametrize(
    "provenance",
    (
        ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1,
        ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    ),
)
def test_policy_history_input_encoding_is_exact_and_provenance_neutral(
    dwell: int,
    expected: int,
    provenance: str,
) -> None:
    available = dwell > 0
    context = ArmHoldHistoryContext(
        previous_mode=np.asarray([1.0 if available else 0.0, 0.0]),
        available_mask=np.asarray([available, available]),
        dwell_steps=np.asarray([dwell, dwell]),
        provenance=provenance,
    )

    encoded = encode_policy_arm_hold_history_input(
        context,
        _input_contract(),
    )

    assert encoded.dtype == np.int64
    assert encoded.shape == (2,)
    assert encoded.tolist() == [expected, expected]
    assert not encoded.flags.writeable


def test_policy_history_input_contract_round_trips_with_hash() -> None:
    contract = _input_contract()
    payload = contract.to_dict(include_hash=True)

    assert PolicyArmHoldHistoryInputContract.from_dict(payload) == contract
    tampered = dict(payload)
    tampered["contract_hash"] = "sha256:wrong"
    with pytest.raises(ArmHoldHistoryContextError, match="does not match"):
        PolicyArmHoldHistoryInputContract.from_dict(tampered)


def test_policy_artifact_history_input_is_additive_and_bound_to_output() -> None:
    legacy = PolicyArtifactContract(policy_io_layout=_layout())
    legacy_payload = legacy.to_dict()
    assert POLICY_ARM_HOLD_HISTORY_INPUT_METADATA_KEY not in legacy_payload
    assert POLICY_ARM_HOLD_HISTORY_INPUT_HASH_KEY not in legacy_payload

    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        arm_hold_output=_output_contract(),
        arm_hold_history_input=_input_contract(),
    )
    payload = artifact.to_dict()
    restored = PolicyArtifactContract.from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.arm_hold_history_input == _input_contract()
    assert len(restored.inputs) == 2


def test_policy_artifact_history_input_fails_closed() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="requires an arm-hold output"):
        PolicyArtifactContract(
            policy_io_layout=_layout(),
            arm_hold_history_input=_input_contract(),
        )

    with pytest.raises(DatasetLayoutBindingError, match="collides"):
        PolicyArtifactContract(
            policy_io_layout=_layout(),
            arm_hold_output=_output_contract(),
            arm_hold_history_input=PolicyArmHoldHistoryInputContract(
                input_key="state"
            ),
        )

    artifact = PolicyArtifactContract(
        policy_io_layout=_layout(),
        arm_hold_output=_output_contract(),
        arm_hold_history_input=_input_contract(),
    )
    payload = artifact.to_dict()
    payload[POLICY_ARM_HOLD_HISTORY_INPUT_HASH_KEY] = "sha256:wrong"
    with pytest.raises(DatasetLayoutBindingError, match="does not match"):
        PolicyArtifactContract.from_dict(payload)
