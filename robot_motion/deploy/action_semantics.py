"""Typed, model-neutral sidecars paired with deployment action keyframes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.policy_action_semantics import (
    PolicyArmHoldOutputContract,
)


@dataclass(frozen=True)
class ArmHoldKeyframePrediction:
    """One action-aligned pair of left/right hold probabilities."""

    probabilities: tuple[float, float]
    contract_hash: str

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.probabilities)
        if len(values) != 2 or not all(
            np.isfinite(value) and 0.0 <= value <= 1.0 for value in values
        ):
            raise ValueError(
                "arm hold probabilities must contain two finite values in [0, 1]."
            )
        contract_hash = str(self.contract_hash).strip()
        if not contract_hash:
            raise ValueError("arm hold prediction contract_hash must be non-empty.")
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "contract_hash", contract_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probabilities": list(self.probabilities),
            "contract_hash": self.contract_hash,
        }


def _nested_semantics_payload(output: Any) -> Mapping[str, Any] | None:
    if isinstance(output, Mapping):
        payload = output.get("action_semantics")
        return payload if isinstance(payload, Mapping) else output
    return None


def _declared_probabilities(
    output: Any,
    *,
    contract: PolicyArmHoldOutputContract,
) -> tuple[Any, Any]:
    probabilities = getattr(output, "arm_hold_probabilities", None)
    contract_hash = getattr(output, "arm_hold_contract_hash", None)
    payload = _nested_semantics_payload(output)
    if payload is not None:
        if probabilities is None:
            probabilities = payload.get(contract.output_key)
        if contract_hash is None:
            contract_hash = (
                payload.get("arm_hold_output_contract_hash")
                or payload.get("contract_hash")
            )
    return probabilities, contract_hash


def bind_arm_hold_predictions(
    output: Any,
    actions: Sequence[Any],
    *,
    contract: PolicyArmHoldOutputContract,
) -> list[Any]:
    """Validate a declared semantic chunk and pair every row with its action.

    ``actions`` are reconstructed through ``dataclasses.replace`` by the
    caller-facing action normalizer, avoiding a second queue or index map.
    """

    from dataclasses import replace

    action_list = list(actions)
    embedded = [getattr(action, "arm_hold_prediction", None) for action in action_list]
    if action_list and all(
        isinstance(value, ArmHoldKeyframePrediction) for value in embedded
    ):
        for value in embedded:
            assert isinstance(value, ArmHoldKeyframePrediction)
            if value.contract_hash != contract.contract_hash:
                raise ValueError(
                    "Embedded arm-hold prediction contract hash mismatch."
                )
        return action_list
    if any(value is not None for value in embedded):
        raise ValueError(
            "Arm-hold predictions must be present on every action keyframe."
        )

    raw_probabilities, raw_contract_hash = _declared_probabilities(
        output,
        contract=contract,
    )
    if raw_probabilities is None:
        raise ValueError(
            f"Policy output is missing declared arm-hold key "
            f"{contract.output_key!r}."
        )
    if str(raw_contract_hash or "") != contract.contract_hash:
        raise ValueError("Policy output arm-hold contract hash mismatch.")
    probabilities = np.asarray(raw_probabilities, dtype=np.float32)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    expected_shape = (len(action_list), len(contract.side_order))
    if probabilities.shape != expected_shape:
        raise ValueError(
            "Arm-hold probability shape must exactly match action horizon and "
            f"side order: expected {expected_shape}, got {probabilities.shape}."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Arm-hold probabilities contain non-finite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "Arm-hold output must contain probabilities in [0, 1], not logits."
        )
    return [
        replace(
            action,
            arm_hold_prediction=ArmHoldKeyframePrediction(
                probabilities=(
                    float(probabilities[index, 0]),
                    float(probabilities[index, 1]),
                ),
                contract_hash=contract.contract_hash,
            ),
        )
        for index, action in enumerate(action_list)
    ]


__all__ = [
    "ArmHoldKeyframePrediction",
    "bind_arm_hold_predictions",
]
